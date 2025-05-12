use anyhow::Result;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use parking_lot::{Mutex, MutexGuard};
use std::sync::Arc;

use super::HookInstance;

pub trait ThunkableClosure<R, T, Args>
where
    R: 'static,
    T: 'static,
    Args: 'static,
    Self: 'static,
{
    fn into_raw_closure(self) -> *const dyn Fn();

    /// Returns the thunk associated with this closure.
    ///
    fn thunk(&self) -> *const ();

    fn unique_name(&self) -> String {
        format!("trampoline_{:?}", std::any::TypeId::of::<Self>())
    }

    /// This is the signature of the rust thunk that is returned from [`Self::thunk`]
    fn thunk_cranelift_sig(&self, module: &mut JITModule) -> cranelift::prelude::Signature;

    /// This is the signature of the original function.
    fn original_cranelift_sig(&self, module: &mut JITModule) -> cranelift::prelude::Signature;

    /// Make a trampoline for this closure. This closure has the address of the thunk and the id
    /// baked in.
    fn make_trampoline(&self, module: &mut JITModule, id: usize) -> Result<*const ()> {
        // Signature of the trampoline that we are going to be swapping into place of the original fn
        let original_sig = { self.original_cranelift_sig(module) };

        // Signature of the rust thunk
        let thunk_sig = { self.thunk_cranelift_sig(module) };

        let mut ctx = module.make_context();
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        ctx.func.signature = original_sig;

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let block = builder.create_block();
            // Make our function params the entry block params
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);
            builder.seal_block(block);

            // Bake in the id and the pointer to thunk.
            let const_id = builder.ins().iconst(types::I64, id as i64);
            let thunk_id = builder.ins().iconst(types::I64, self.thunk() as i64);

            // Build up the params that we are going to pass to the thunk
            // This is the id, followed by this, followed by $($Args,)*
            let params = builder.block_params(block);
            let mut call_args = vec![const_id];
            call_args.extend_from_slice(params);

            // Call the thunk
            let thunk_sig = builder.import_signature(thunk_sig);
            let call = builder.ins().call_indirect(thunk_sig, thunk_id, &call_args);

            // Return whatever thunk returns
            let result = builder.inst_results(call)[0];
            builder.ins().return_(&[result]);
        }

        let trampoline_id =
            module.declare_function(&self.unique_name(), Linkage::Export, &ctx.func.signature)?;

        module.define_function(trampoline_id, &mut ctx)?;

        module.clear_context(&mut ctx);
        module.finalize_definitions()?;

        let code_ptr = module.get_finalized_function(trampoline_id);
        Ok(unsafe { std::mem::transmute::<*const u8, *const ()>(code_ptr) })
    }
}

use paste::paste;

/// [`Call`] is implemented for `_FuncContext<A...>`.
pub trait Call<R: 'static, T: 'static, Args: 'static> {
    fn call(&self, this: &mut T, args: Args) -> R;
}

/// impl_func implements [`ThunkableClosure`] for any number of parameters.
/// In addition it also creates some other types and structs that are used in the call to the closure:
///
/// * `_FuncContext<A...>` which holds the hook instance and the original function pointer.
///
/// It also implements [`Call`] for the `_FuncContext<A...>` type, such that you can pass it to
///  [`call_original`].
///
/// ```rs
/// // ...
/// |ctx, this, a: usize, b: usize| -> usize {
///     call_original(ctx, this, (a, b))
/// }
/// // ...
/// ```
///
/// This is loosely based on the
/// [`NativeFunc` implementation from Rhai](https://github.com/rhaiscript/rhai/blob/main/src/func/register.rs#L81).
///
macro_rules! impl_func {
    ($($args:ident: $Args:ident),*) => {
        paste! {
            type [<_RawFunc $($Args )*>]<TRet, TThis, $($Args,)*> =
                unsafe extern "C" fn (*mut TThis, $($Args,)*) -> TRet;

            pub struct [<_FuncContext $($Args )*>]<TRet, TThis, $($Args,)*> {
                pub original_fn: [<_RawFunc $($Args )*>]<TRet, TThis, $($Args,)*>,
                #[allow(dead_code)]
                hook_instance: Arc<Mutex<HookInstance>>,
            }

            impl<
                TRet,
                TThis,
                $($Args,)*
            >
            Call<TRet, TThis, ($($Args,)*)> for &[<_FuncContext $($Args )*>]<TRet, TThis, $($Args,)*>
            where
                TRet: 'static,
                TThis: 'static,
                $($Args: 'static,)*
            {
                fn call(
                    &self,
                    this: &mut TThis,
                    args: ($($Args,)*)
                ) -> TRet {
                    let ($($args,)*) = args;
                    unsafe { (self.original_fn)(this as *mut TThis, $($args,)*) }
                }
            }

            impl<
                'ctx,
                'this,
                TClosure,
                TRet,
                TThis,
                $($Args,)*
            > ThunkableClosure<TRet, TThis, ($($Args,)*)> for TClosure
            where
                TRet: 'static + AsCraneliftAbi,
                TThis: 'static,
                &'this mut TThis: AsCraneliftAbi,
                $($Args: 'static + AsCraneliftAbi,)*
                TClosure: (
                    Fn(
                        &'ctx [<_FuncContext $($Args )*>]<TRet, TThis, $($Args,)*>,
                        &'this mut TThis,
                        $($Args,)*
                    ) -> TRet
                ) + Send + Sync + 'static,
            {
                fn thunk(&self) -> *const () {
                    unsafe extern "C" fn func<TRet, TThis, $($Args,)*>(
                        index: usize,
                        this: *mut TThis,
                        $($args: $Args,)*
                    ) -> TRet
                    where
                        TRet: 'static,
                        TThis: 'static,
                        $(
                            $Args: 'static
                        ),*
                    {
                        // Get the closure, original_fn and instance
                        let (
                                closure,
                                original_function,
                                hook_instance,
                            ): (
                                // Closure trait type, should match TClosure above but with dyn
                                // TODO(emily): You could (and should) be passing down the lifetimes from the outer
                                // scope to here.
                                *const (
                                    dyn Fn(
                                        &[<_FuncContext $($Args )*>]<TRet, TThis, $($Args,)*>,
                                        &mut TThis,
                                        $($Args,)*
                                    ) -> TRet + Send + Sync + 'static
                                ),

                                // Original function type
                                [<_RawFunc $($Args )*>]<TRet, TThis, $($Args,)*>,

                                // Hook instance
                                Arc<Mutex<HookInstance>>,
                            ) = {
                            let hook_instance = HookInstance::for_instance(this as *mut ());
                            let h = hook_instance.lock();
                            let closure = h.closure(index);
                            let original_function = h.original_function(index);

                            drop(h);

                            #[allow(clippy::missing_transmute_annotations)]
                            (
                                std::mem::transmute(closure),
                                std::mem::transmute(original_function),
                                hook_instance,
                            )
                        };

                        let context = [<_FuncContext $($Args )*>]::<TRet, TThis, $($Args,)*> {
                            original_fn: original_function,
                            hook_instance
                        };

                        (*closure)(&context, &mut *this, $($args,)*)
                    }

                    func::<TRet, TThis, $($Args,)*> as *const ()
                }

                fn thunk_cranelift_sig(&self, module: &mut JITModule) -> cranelift::prelude::Signature {
                    let mut signature = module.make_signature();
                    signature.returns.push(cranelift_abi::<TRet>());

                    signature.params.push(AbiParam::new(types::I64));
                    signature.params.push(cranelift_abi::<*mut TThis>());
                    $(
                        signature.params.push(cranelift_abi::<$Args>());
                    )*
                    signature
                }

                fn original_cranelift_sig(&self, module: &mut JITModule) -> cranelift::prelude::Signature {
                    let mut signature = module.make_signature();
                    signature.returns.push(cranelift_abi::<TRet>());

                    signature.params.push(cranelift_abi::<*mut TThis>());
                    $(
                        signature.params.push(cranelift_abi::<$Args>());
                    )*
                    signature
                }

                fn into_raw_closure(self) -> *const dyn Fn() {
                    let b = Box::new(self);
                    let b = b as Box<dyn Fn(
                        &'ctx [<_FuncContext $($Args )*>]<TRet, TThis, $($Args,)*>,
                        &'this mut TThis,
                        $($Args,)*
                    ) -> TRet + Send + Sync + 'static>;

                    unsafe { std::mem::transmute(Box::into_raw(b)) }
                }
            }
        }
    };
}

impl_func!();
impl_func!(a: A);
impl_func!(a: A, b: B);
impl_func!(a: A, b: B, c: C);
impl_func!(a: A, b: B, c: C, d: D);
impl_func!(a: A, b: B, c: C, d: D, e: E);

/// Call the original function for a context.
pub fn call_original<TRet: 'static, TThis: 'static, TArgs: 'static>(
    ctx: impl Call<TRet, TThis, TArgs>,
    this: &mut TThis,
    args: TArgs,
) -> TRet {
    ctx.call(this, args)
}

pub(super) struct TrampolineStorage {
    module: Mutex<JITModule>,
}

impl TrampolineStorage {
    pub(super) fn new() -> Result<Self> {
        let builder = JITBuilder::new(cranelift_module::default_libcall_names())?;
        let module = JITModule::new(builder);
        Ok(Self {
            module: Mutex::new(module),
        })
    }

    pub(super) fn module(&self) -> MutexGuard<JITModule> {
        self.module.lock()
    }
}

impl Drop for TrampolineStorage {
    fn drop(&mut self) {
        // TODO(emily): You should probably call free_memory here to not leak memory.
        // let module = self.module.lock();
        // module.free_memory();
    }
}

pub trait AsCraneliftAbi {
    fn as_cranelift_abi() -> AbiParam;
}

/// Get the cranelift abi for a type T
fn cranelift_abi<T: AsCraneliftAbi>() -> AbiParam {
    T::as_cranelift_abi()
}

// Implement AsCraneliftAbi for a bunch of types

impl<T> AsCraneliftAbi for *const T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(types::I64)
    }
}

impl<T> AsCraneliftAbi for *mut T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(types::I64)
    }
}

impl<T> AsCraneliftAbi for &mut T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(types::I64)
    }
}

impl<T> AsCraneliftAbi for &T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(types::I64)
    }
}

macro_rules! cranelift_abi {
    ($t:ty, $abi:ident) => {
        impl AsCraneliftAbi for $t {
            fn as_cranelift_abi() -> AbiParam {
                AbiParam::new(types::$abi)
            }
        }
    };
}

cranelift_abi!(f32, F32);
cranelift_abi!(f64, F64);

cranelift_abi!(usize, I64);
cranelift_abi!(isize, I64);

cranelift_abi!(u32, I32);
cranelift_abi!(i32, I32);

cranelift_abi!(u16, I16);
cranelift_abi!(i16, I16);

cranelift_abi!(u8, I8);
cranelift_abi!(i8, I8);
