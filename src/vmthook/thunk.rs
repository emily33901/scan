use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use parking_lot::Mutex;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        OnceLock,
    },
};

use super::{HookDiagnostics, HookId, HookStatus};

#[derive(Debug)]
pub(super) enum TrampolineError {
    Initialization(String),
    Compilation(String),
    Finalization(String),
}

impl std::fmt::Display for TrampolineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initialization(message)
            | Self::Compilation(message)
            | Self::Finalization(message) => formatter.write_str(message),
        }
    }
}

mod sealed {
    pub trait Abi {}
    pub trait PreHook<R, T, Args> {}
}

#[doc(hidden)]
pub trait DispatchControl: Send + Sync {
    fn context_address(&self) -> *const ();
    fn dispatcher_address(&self) -> *const ();
    fn thunk_cranelift_sig(&self, module: &mut JITModule) -> Signature;
    fn original_cranelift_sig(&self, module: &mut JITModule) -> Signature;
    fn set_enabled(&self, enabled: bool) -> bool;
    fn retire(&self, status: HookStatus);
    fn diagnostics(&self) -> HookDiagnostics;
}

#[doc(hidden)]
pub trait ThunkablePreHook<R, T, Args>:
    sealed::PreHook<R, T, Args> + Send + Sync + 'static
where
    R: 'static,
    T: 'static,
    Args: 'static,
{
    unsafe fn into_dispatch(self, original: *const (), hook_id: HookId)
        -> Box<dyn DispatchControl>;
}

fn increment_saturated(counter: &AtomicU64) {
    // Diagnostics are deliberately best-effort: dispatch must never spin under
    // contention merely to update a counter. A lost sample is preferable to an
    // unbounded retry on the host's real-time thread.
    let value = counter.load(Ordering::Relaxed);
    if value != u64::MAX {
        let _ =
            counter.compare_exchange_weak(value, value + 1, Ordering::Relaxed, Ordering::Relaxed);
    }
}

struct BusyGuard<'a>(&'a AtomicBool);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

struct DispatchState {
    hook_id: HookId,
    run_state: AtomicU8,
    busy: AtomicBool,
    invocations: AtomicU64,
    bypasses: AtomicU64,
    panics: AtomicU64,
    last_status: AtomicU8,
}

impl DispatchState {
    const ENABLED: u8 = 0;
    const DISABLED: u8 = 1;
    const PANICKED: u8 = 2;
    const RETIRED: u8 = 3;
    const RESTORED: u8 = 4;

    fn new(hook_id: HookId) -> Self {
        Self {
            hook_id,
            run_state: AtomicU8::new(Self::ENABLED),
            busy: AtomicBool::new(false),
            invocations: AtomicU64::new(0),
            bypasses: AtomicU64::new(0),
            panics: AtomicU64::new(0),
            last_status: AtomicU8::new(HookStatus::Ready as u8),
        }
    }

    fn diagnostics(&self) -> HookDiagnostics {
        HookDiagnostics {
            hook_id: self.hook_id,
            enabled: self.run_state.load(Ordering::Acquire) == Self::ENABLED,
            invocations: self.invocations.load(Ordering::Relaxed),
            bypasses: self.bypasses.load(Ordering::Relaxed),
            panics: self.panics.load(Ordering::Relaxed),
            last_status: HookStatus::from_u8(self.last_status.load(Ordering::Relaxed)),
        }
    }

    fn set_enabled(&self, enabled: bool) -> bool {
        if enabled {
            self.run_state
                .compare_exchange(
                    Self::DISABLED,
                    Self::ENABLED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
                || self.run_state.load(Ordering::Acquire) == Self::ENABLED
        } else {
            loop {
                let state = self.run_state.load(Ordering::Acquire);
                if state == Self::PANICKED
                    || state == Self::RETIRED
                    || state == Self::RESTORED
                    || state == Self::DISABLED
                {
                    return true;
                }
                if self
                    .run_state
                    .compare_exchange(
                        Self::ENABLED,
                        Self::DISABLED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return true;
                }
            }
        }
    }

    fn inactive_status(&self) -> HookStatus {
        match self.run_state.load(Ordering::Acquire) {
            Self::PANICKED => HookStatus::Panicked,
            Self::RESTORED => HookStatus::Restored,
            _ => HookStatus::Disabled,
        }
    }
}

use paste::paste;

macro_rules! impl_func {
    ($($args:ident: $Args:ident),*) => {
        paste! {
            type [<_RawFunc $($Args )*>]<TRet, TThis, $($Args,)*> =
                unsafe extern "C" fn(*mut TThis, $($Args,)*) -> TRet;

            struct [<_DispatchContext $($Args )*>]<TClosure, TRet, TThis, $($Args,)*> {
                state: DispatchState,
                original_fn: [<_RawFunc $($Args )*>]<TRet, TThis, $($Args,)*>,
                handler: TClosure,
            }

            impl<TClosure, TRet, TThis, $($Args,)*> DispatchControl
                for [<_DispatchContext $($Args )*>]<TClosure, TRet, TThis, $($Args,)*>
            where
                TRet: 'static + AsCraneliftAbi,
                TThis: 'static,
                $($Args: Copy + 'static + AsCraneliftAbi,)*
                TClosure: Fn(&TThis, $($Args,)*) + Send + Sync + 'static,
            {
                fn context_address(&self) -> *const () {
                    (self as *const Self).cast()
                }

                fn dispatcher_address(&self) -> *const () {
                    unsafe extern "C" fn dispatch<TClosure, TRet, TThis, $($Args,)*>(
                        context: *const [<_DispatchContext $($Args )*>]<TClosure, TRet, TThis, $($Args,)*>,
                        this: *mut TThis,
                        $($args: $Args,)*
                    ) -> TRet
                    where
                        TThis: 'static,
                        $($Args: Copy + 'static,)*
                        TClosure: Fn(&TThis, $($Args,)*) + Send + Sync + 'static,
                    {
                        let context = unsafe { &*context };
                        increment_saturated(&context.state.invocations);
                        if context.state.run_state.load(Ordering::Acquire) != DispatchState::ENABLED {
                            increment_saturated(&context.state.bypasses);
                            context.state.last_status.store(context.state.inactive_status() as u8, Ordering::Relaxed);
                            return unsafe { (context.original_fn)(this, $($args,)*) };
                        }
                        if context.state.busy.compare_exchange(
                            false, true, Ordering::Acquire, Ordering::Relaxed,
                        ).is_err() {
                            increment_saturated(&context.state.bypasses);
                            context.state.last_status.store(HookStatus::BusyBypass as u8, Ordering::Relaxed);
                            return unsafe { (context.original_fn)(this, $($args,)*) };
                        }

                        // Keep exclusivity through the original call so original-driven reentry bypasses.
                        let _busy = BusyGuard(&context.state.busy);
                        if context.state.run_state.load(Ordering::Acquire) != DispatchState::ENABLED {
                            increment_saturated(&context.state.bypasses);
                            context.state.last_status.store(context.state.inactive_status() as u8, Ordering::Relaxed);
                            return unsafe { (context.original_fn)(this, $($args,)*) };
                        }
                        context.state.last_status.store(HookStatus::Running as u8, Ordering::Relaxed);
                        // DEFERRED(vmthook): expose a raw receiver before supporting concurrent
                        // handlers; `&TThis` relies on the current exclusive, non-mutating contract.
                        let handler_result = catch_unwind(AssertUnwindSafe(|| {
                            (context.handler)(unsafe { &*this }, $($args,)*);
                        }));
                        if let Err(payload) = handler_result {
                            increment_saturated(&context.state.panics);
                            context.state.run_state.store(DispatchState::PANICKED, Ordering::Release);
                            context.state.last_status.store(HookStatus::Panicked as u8, Ordering::Relaxed);
                            // A panic payload may itself panic while being dropped. Never let
                            // that second panic cross the generated extern boundary.
                            std::mem::forget(payload);
                        } else {
                            context.state.last_status.store(HookStatus::Ready as u8, Ordering::Relaxed);
                        }

                        // The pre-hook cannot suppress or duplicate the captured original call.
                        unsafe { (context.original_fn)(this, $($args,)*) }
                    }

                    dispatch::<TClosure, TRet, TThis, $($Args,)*> as *const ()
                }

                fn thunk_cranelift_sig(&self, module: &mut JITModule) -> Signature {
                    let mut signature = module.make_signature();
                    signature.returns.push(cranelift_abi::<TRet>());
                    signature.params.push(AbiParam::new(module.target_config().pointer_type()));
                    signature.params.push(cranelift_abi::<*mut TThis>());
                    $(signature.params.push(cranelift_abi::<$Args>());)*
                    signature
                }

                fn original_cranelift_sig(&self, module: &mut JITModule) -> Signature {
                    let mut signature = module.make_signature();
                    signature.returns.push(cranelift_abi::<TRet>());
                    signature.params.push(cranelift_abi::<*mut TThis>());
                    $(signature.params.push(cranelift_abi::<$Args>());)*
                    signature
                }

                fn set_enabled(&self, enabled: bool) -> bool { self.state.set_enabled(enabled) }
                fn retire(&self, status: HookStatus) {
                    let run_state = if status == HookStatus::Restored {
                        DispatchState::RESTORED
                    } else {
                        DispatchState::RETIRED
                    };
                    self.state.run_state.store(run_state, Ordering::Release);
                    self.state.last_status.store(status as u8, Ordering::Relaxed);
                }
                fn diagnostics(&self) -> HookDiagnostics { self.state.diagnostics() }
            }

            impl<TClosure, TRet, TThis, $($Args,)*>
                ThunkablePreHook<TRet, TThis, ($($Args,)*)> for TClosure
            where
                TRet: 'static + AsCraneliftAbi,
                TThis: 'static,
                $($Args: Copy + 'static + AsCraneliftAbi,)*
                TClosure: Fn(&TThis, $($Args,)*) + Send + Sync + 'static,
            {
                unsafe fn into_dispatch(
                    self,
                    original: *const (),
                    hook_id: HookId,
                ) -> Box<dyn DispatchControl> {
                    Box::new([<_DispatchContext $($Args )*>] {
                        state: DispatchState::new(hook_id),
                        original_fn: unsafe {
                            std::mem::transmute::<
                                *const (),
                                [<_RawFunc $($Args )*>]<TRet, TThis, $($Args,)*>,
                            >(original)
                        },
                        handler: self,
                    })
                }
            }

            impl<TClosure, TRet, TThis, $($Args,)*>
                sealed::PreHook<TRet, TThis, ($($Args,)*)> for TClosure
            where
                TRet: 'static + AsCraneliftAbi,
                TThis: 'static,
                $($Args: Copy + 'static + AsCraneliftAbi,)*
                TClosure: Fn(&TThis, $($Args,)*) + Send + Sync + 'static,
            {}
        }
    };
}

impl_func!();
impl_func!(a: A);
impl_func!(a: A, b: B);
impl_func!(a: A, b: B, c: C);
impl_func!(a: A, b: B, c: C, d: D);
impl_func!(a: A, b: B, c: C, d: D, e: E);

pub(super) struct TrampolineStorage {
    module: Mutex<JITModule>,
}

impl TrampolineStorage {
    fn new() -> std::result::Result<Self, TrampolineError> {
        let builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|error| TrampolineError::Initialization(error.to_string()))?;
        Ok(Self {
            module: Mutex::new(JITModule::new(builder)),
        })
    }

    pub(super) fn global() -> std::result::Result<&'static Self, TrampolineError> {
        // DEFERRED(vmthook): validate Cranelift executable-memory allocation and
        // MAP_JIT entitlements inside the signed production Ableton host on macOS.
        static STORAGE: OnceLock<std::result::Result<TrampolineStorage, String>> = OnceLock::new();
        match STORAGE.get_or_init(|| Self::new().map_err(|error| error.to_string())) {
            Ok(storage) => Ok(storage),
            Err(error) => Err(TrampolineError::Initialization(error.clone())),
        }
    }

    pub(super) fn make_trampoline(
        &self,
        dispatch: &dyn DispatchControl,
        hook_id: HookId,
    ) -> std::result::Result<*const (), TrampolineError> {
        let mut module = self.module.lock();
        let original_sig = dispatch.original_cranelift_sig(&mut module);
        let thunk_sig = dispatch.thunk_cranelift_sig(&mut module);
        let pointer_type = module.target_config().pointer_type();
        let mut ctx = module.make_context();
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        ctx.func.signature = original_sig;
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);
            builder.seal_block(block);
            let context = builder
                .ins()
                .iconst(pointer_type, dispatch.context_address() as i64);
            let dispatcher = builder
                .ins()
                .iconst(pointer_type, dispatch.dispatcher_address() as i64);
            let mut call_args = Vec::with_capacity(builder.block_params(block).len() + 1);
            call_args.push(context);
            call_args.extend_from_slice(builder.block_params(block));
            let signature = builder.import_signature(thunk_sig);
            let call = builder
                .ins()
                .call_indirect(signature, dispatcher, &call_args);
            let results = builder.inst_results(call).to_vec();
            builder.ins().return_(&results);
        }
        let name = format!("vmthook_{}", hook_id.0);
        let function = module
            .declare_function(&name, Linkage::Local, &ctx.func.signature)
            .map_err(|error| TrampolineError::Compilation(error.to_string()))?;
        module
            .define_function(function, &mut ctx)
            .map_err(|error| TrampolineError::Compilation(error.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|error| TrampolineError::Finalization(error.to_string()))?;
        Ok(module.get_finalized_function(function).cast())
    }
}

// DEFERRED(vmthook): support general around-hooks after defining original-call and
// panic semantics that cannot accidentally invoke the original twice.
// DEFERRED(vmthook): add allow/blocking reentrancy policies after removing `&TThis`.
// DEFERRED(vmthook): foreign exceptions and native faults are not contained here.
// DEFERRED(vmthook): reject aggregate, vector, variadic, and hidden-sret ABIs explicitly.

pub trait AsCraneliftAbi: sealed::Abi {
    fn as_cranelift_abi() -> AbiParam;
}

fn cranelift_abi<T: AsCraneliftAbi>() -> AbiParam {
    T::as_cranelift_abi()
}

fn pointer_type() -> Type {
    if cfg!(target_pointer_width = "64") {
        types::I64
    } else {
        types::I32
    }
}

impl<T> AsCraneliftAbi for *const T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(pointer_type())
    }
}
impl<T> sealed::Abi for *const T {}
impl<T> AsCraneliftAbi for *mut T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(pointer_type())
    }
}
impl<T> sealed::Abi for *mut T {}
impl<T> AsCraneliftAbi for &mut T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(pointer_type())
    }
}
impl<T> sealed::Abi for &mut T {}
impl<T> AsCraneliftAbi for &T {
    fn as_cranelift_abi() -> AbiParam {
        AbiParam::new(pointer_type())
    }
}
impl<T> sealed::Abi for &T {}

macro_rules! cranelift_abi {
    ($t:ty, $abi:ident) => {
        impl AsCraneliftAbi for $t {
            fn as_cranelift_abi() -> AbiParam {
                AbiParam::new(types::$abi)
            }
        }
        impl sealed::Abi for $t {}
    };
}

cranelift_abi!(f32, F32);
cranelift_abi!(f64, F64);
#[cfg(target_pointer_width = "64")]
cranelift_abi!(usize, I64);
#[cfg(target_pointer_width = "64")]
cranelift_abi!(isize, I64);
#[cfg(target_pointer_width = "32")]
cranelift_abi!(usize, I32);
#[cfg(target_pointer_width = "32")]
cranelift_abi!(isize, I32);
cranelift_abi!(u32, I32);
cranelift_abi!(i32, I32);
macro_rules! narrow_cranelift_abi {
    ($t:ty, $abi:ident, $extension:ident) => {
        impl AsCraneliftAbi for $t {
            fn as_cranelift_abi() -> AbiParam {
                AbiParam::new(types::$abi).$extension()
            }
        }
        impl sealed::Abi for $t {}
    };
}

narrow_cranelift_abi!(u16, I16, uext);
narrow_cranelift_abi!(i16, I16, sext);
narrow_cranelift_abi!(u8, I8, uext);
narrow_cranelift_abi!(i8, I8, sext);

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::ptr;
    use std::sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc, Barrier,
    };

    struct ThreadTrackingAllocator;

    thread_local! {
        static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for ThreadTrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            TRACK_ALLOCATIONS.with(|tracking| {
                if tracking.get() {
                    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            TRACK_ALLOCATIONS.with(|tracking| {
                if tracking.get() {
                    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            TRACK_ALLOCATIONS.with(|tracking| {
                if tracking.get() {
                    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.realloc(pointer, layout, size) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[global_allocator]
    static TEST_ALLOCATOR: ThreadTrackingAllocator = ThreadTrackingAllocator;

    fn allocations_during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATION_COUNT.with(|count| count.set(0));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        let result = operation();
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        let allocations = ALLOCATION_COUNT.with(Cell::get);
        (result, allocations)
    }

    struct Target {
        original_calls: AtomicU32,
    }

    unsafe extern "C" fn original(this: *mut Target, argument: u32) -> u32 {
        unsafe { &*this }
            .original_calls
            .fetch_add(1, Ordering::Relaxed);
        argument + 1
    }

    type TestDispatcher = unsafe extern "C" fn(*const (), *mut Target, u32) -> u32;

    unsafe fn make_dispatch<F>(handler: F, id: u64) -> Box<dyn DispatchControl>
    where
        F: ThunkablePreHook<u32, Target, (u32,)>,
    {
        unsafe { handler.into_dispatch(original as *const (), HookId(id)) }
    }

    unsafe fn direct_dispatch(
        dispatch: &dyn DispatchControl,
        target: *mut Target,
        argument: u32,
    ) -> u32 {
        let dispatcher: TestDispatcher =
            unsafe { std::mem::transmute(dispatch.dispatcher_address()) };
        unsafe { dispatcher(dispatch.context_address(), target, argument) }
    }

    #[test]
    fn direct_dispatch_calls_handler_then_original() {
        let handler_calls = Arc::new(AtomicU32::new(0));
        let observed = Arc::clone(&handler_calls);
        let dispatch = unsafe {
            make_dispatch(
                move |_this: &Target, argument: u32| {
                    observed.store(argument, Ordering::Relaxed);
                },
                10,
            )
        };
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };

        let result = unsafe { direct_dispatch(&*dispatch, &mut target, 41) };

        assert_eq!(result, 42);
        assert_eq!(handler_calls.load(Ordering::Relaxed), 41);
        assert_eq!(target.original_calls.load(Ordering::Relaxed), 1);
        assert_eq!(dispatch.diagnostics().invocations, 1);
    }

    #[test]
    fn panic_disables_handler_and_calls_original_once() {
        let handler_calls = Arc::new(AtomicU32::new(0));
        let observed = Arc::clone(&handler_calls);
        let dispatch = unsafe {
            make_dispatch(
                move |_this: &Target, _argument: u32| {
                    observed.fetch_add(1, Ordering::Relaxed);
                    panic!("expected test panic");
                },
                11,
            )
        };
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };

        assert_eq!(unsafe { direct_dispatch(&*dispatch, &mut target, 1) }, 2);
        assert_eq!(unsafe { direct_dispatch(&*dispatch, &mut target, 2) }, 3);

        let diagnostics = dispatch.diagnostics();
        assert!(!diagnostics.enabled);
        assert_eq!(diagnostics.panics, 1);
        assert_eq!(diagnostics.bypasses, 1);
        assert_eq!(handler_calls.load(Ordering::Relaxed), 1);
        assert_eq!(target.original_calls.load(Ordering::Relaxed), 2);

        assert!(!dispatch.set_enabled(true));
        assert_eq!(unsafe { direct_dispatch(&*dispatch, &mut target, 3) }, 4);
        assert_eq!(handler_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn panic_payload_destructor_cannot_escape_dispatch() {
        struct PanicsOnDrop;
        impl Drop for PanicsOnDrop {
            fn drop(&mut self) {
                panic!("panic payload destructor");
            }
        }

        let dispatch = unsafe {
            make_dispatch(
                |_this: &Target, _argument: u32| std::panic::panic_any(PanicsOnDrop),
                15,
            )
        };
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };

        assert_eq!(unsafe { direct_dispatch(&*dispatch, &mut target, 7) }, 8);
        assert_eq!(target.original_calls.load(Ordering::Relaxed), 1);
        assert_eq!(dispatch.diagnostics().panics, 1);
    }

    #[test]
    fn concurrent_entry_bypasses_without_waiting() {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let handler_calls = Arc::new(AtomicU32::new(0));
        let handler_entered = Arc::clone(&entered);
        let handler_release = Arc::clone(&release);
        let observed = Arc::clone(&handler_calls);
        let dispatch = unsafe {
            make_dispatch(
                move |_this: &Target, _argument: u32| {
                    observed.fetch_add(1, Ordering::Relaxed);
                    handler_entered.wait();
                    handler_release.wait();
                },
                12,
            )
        };
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };
        let context_address = dispatch.context_address() as usize;
        let dispatcher_address = dispatch.dispatcher_address() as usize;
        let target_address = (&mut target as *mut Target) as usize;

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let dispatcher: TestDispatcher = unsafe { std::mem::transmute(dispatcher_address) };
                unsafe {
                    dispatcher(
                        context_address as *const (),
                        target_address as *mut Target,
                        10,
                    )
                };
            });
            entered.wait();
            assert_eq!(
                unsafe { direct_dispatch(&*dispatch, target_address as *mut Target, 20) },
                21
            );
            release.wait();
        });

        assert_eq!(handler_calls.load(Ordering::Relaxed), 1);
        assert_eq!(target.original_calls.load(Ordering::Relaxed), 2);
        assert_eq!(dispatch.diagnostics().bypasses, 1);
    }

    #[test]
    fn nested_entry_bypasses_handler_and_calls_original() {
        let context_address = Arc::new(AtomicUsize::new(0));
        let dispatcher_address = Arc::new(AtomicUsize::new(0));
        let handler_calls = Arc::new(AtomicU32::new(0));
        let nested_result = Arc::new(AtomicU32::new(0));
        let captured_context = Arc::clone(&context_address);
        let captured_dispatcher = Arc::clone(&dispatcher_address);
        let observed_calls = Arc::clone(&handler_calls);
        let observed_result = Arc::clone(&nested_result);
        let handler = move |this: &Target, _argument: u32| {
            observed_calls.fetch_add(1, Ordering::Relaxed);
            let dispatcher: TestDispatcher =
                unsafe { std::mem::transmute(captured_dispatcher.load(Ordering::Relaxed)) };
            let result = unsafe {
                dispatcher(
                    captured_context.load(Ordering::Relaxed) as *const (),
                    this as *const Target as *mut Target,
                    40,
                )
            };
            observed_result.store(result, Ordering::Relaxed);
        };
        let dispatch = unsafe { make_dispatch(handler, 17) };
        context_address.store(dispatch.context_address() as usize, Ordering::Relaxed);
        dispatcher_address.store(dispatch.dispatcher_address() as usize, Ordering::Relaxed);
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };

        assert_eq!(unsafe { direct_dispatch(&*dispatch, &mut target, 1) }, 2);
        assert_eq!(nested_result.load(Ordering::Relaxed), 41);
        assert_eq!(handler_calls.load(Ordering::Relaxed), 1);
        assert_eq!(target.original_calls.load(Ordering::Relaxed), 2);
        assert_eq!(dispatch.diagnostics().bypasses, 1);
    }

    #[test]
    fn warmed_trampoline_dispatch_makes_no_framework_allocations() {
        let dispatch = unsafe { make_dispatch(|_: &Target, _: u32| {}, 18) };
        let trampoline = TrampolineStorage::global()
            .unwrap()
            .make_trampoline(&*dispatch, HookId(18))
            .unwrap();
        let function: unsafe extern "C" fn(*mut Target, u32) -> u32 =
            unsafe { std::mem::transmute(trampoline) };
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };
        assert_eq!(unsafe { function(&mut target, 0) }, 1);

        let (result, allocations) = allocations_during(|| unsafe { function(&mut target, 41) });

        assert_eq!(result, 42);
        assert_eq!(allocations, 0);
    }

    fn counting_handler(counter: Arc<AtomicU32>) -> impl Fn(&Target, u32) {
        move |_this, argument| {
            counter.store(argument, Ordering::Relaxed);
        }
    }

    #[test]
    fn jit_trampolines_bake_distinct_contexts() {
        let first_calls = Arc::new(AtomicU32::new(0));
        let second_calls = Arc::new(AtomicU32::new(0));
        let first = unsafe { make_dispatch(counting_handler(Arc::clone(&first_calls)), 13) };
        let second = unsafe { make_dispatch(counting_handler(Arc::clone(&second_calls)), 14) };
        let storage = TrampolineStorage::global().unwrap();
        let first_trampoline = storage.make_trampoline(&*first, HookId(13)).unwrap();
        let second_trampoline = storage.make_trampoline(&*second, HookId(14)).unwrap();
        let first_function: unsafe extern "C" fn(*mut Target, u32) -> u32 =
            unsafe { std::mem::transmute(first_trampoline) };
        let second_function: unsafe extern "C" fn(*mut Target, u32) -> u32 =
            unsafe { std::mem::transmute(second_trampoline) };
        let mut target = Target {
            original_calls: AtomicU32::new(0),
        };

        assert_eq!(unsafe { first_function(&mut target, 30) }, 31);
        assert_eq!(unsafe { second_function(&mut target, 40) }, 41);
        assert_eq!(first_calls.load(Ordering::Relaxed), 30);
        assert_eq!(second_calls.load(Ordering::Relaxed), 40);
        assert_eq!(target.original_calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn diagnostic_counters_saturate() {
        let counter = AtomicU64::new(u64::MAX);
        increment_saturated(&counter);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn jit_preserves_narrow_signed_arguments_and_returns() {
        struct NarrowTarget;
        unsafe extern "C" fn narrow_original(_: *mut NarrowTarget, value: i8) -> i8 {
            value.wrapping_sub(1)
        }
        let handler = |_: &NarrowTarget, _: i8| {};
        let context: Box<dyn DispatchControl> = unsafe {
            <_ as ThunkablePreHook<i8, NarrowTarget, (i8,)>>::into_dispatch(
                handler,
                narrow_original as *const (),
                HookId(16),
            )
        };
        let trampoline = TrampolineStorage::global()
            .unwrap()
            .make_trampoline(&*context, HookId(16))
            .unwrap();
        let function: unsafe extern "C" fn(*mut NarrowTarget, i8) -> i8 =
            unsafe { std::mem::transmute(trampoline) };
        let mut target = NarrowTarget;

        assert_eq!(unsafe { function(&mut target, -100) }, -101);
    }

    #[test]
    fn jit_round_trips_advertised_scalar_and_pointer_types() {
        struct AbiTarget;

        macro_rules! assert_round_trip {
            ($id:expr, $type:ty, $value:expr) => {{
                unsafe extern "C" fn original(_: *mut AbiTarget, value: $type) -> $type {
                    value
                }
                let handler = |_: &AbiTarget, _: $type| {};
                let context: Box<dyn DispatchControl> = unsafe {
                    <_ as ThunkablePreHook<$type, AbiTarget, ($type,)>>::into_dispatch(
                        handler,
                        original as *const (),
                        HookId($id),
                    )
                };
                let trampoline = TrampolineStorage::global()
                    .unwrap()
                    .make_trampoline(&*context, HookId($id))
                    .unwrap();
                let function: unsafe extern "C" fn(*mut AbiTarget, $type) -> $type =
                    unsafe { std::mem::transmute(trampoline) };
                let mut target = AbiTarget;
                assert_eq!(unsafe { function(&mut target, $value) }, $value);
            }};
        }

        assert_round_trip!(100, u8, 201);
        assert_round_trip!(101, i8, -101);
        assert_round_trip!(102, u16, 50_001);
        assert_round_trip!(103, i16, -20_001);
        assert_round_trip!(104, u32, 3_000_000_001);
        assert_round_trip!(105, i32, -2_000_000_001);
        assert_round_trip!(106, usize, usize::MAX / 3);
        assert_round_trip!(107, isize, isize::MIN / 3);
        assert_round_trip!(108, f32, -123.25);
        assert_round_trip!(109, f64, 987_654.125);
        assert_round_trip!(110, *const u8, ptr::without_provenance(0x1234));
        assert_round_trip!(111, *mut u8, ptr::without_provenance_mut(0x5678));
    }

    #[test]
    fn jit_round_trips_zero_and_five_argument_arities() {
        struct AbiTarget {
            observed: AtomicUsize,
        }

        unsafe extern "C" fn zero_original(_: *mut AbiTarget) -> isize {
            -77
        }
        let zero_handler = |_: &AbiTarget| {};
        let zero_context: Box<dyn DispatchControl> = unsafe {
            <_ as ThunkablePreHook<isize, AbiTarget, ()>>::into_dispatch(
                zero_handler,
                zero_original as *const (),
                HookId(112),
            )
        };
        let zero_trampoline = TrampolineStorage::global()
            .unwrap()
            .make_trampoline(&*zero_context, HookId(112))
            .unwrap();
        let zero_function: unsafe extern "C" fn(*mut AbiTarget) -> isize =
            unsafe { std::mem::transmute(zero_trampoline) };
        let mut target = AbiTarget {
            observed: AtomicUsize::new(0),
        };
        assert_eq!(unsafe { zero_function(&mut target) }, -77);

        unsafe extern "C" fn mixed_original(
            _: *mut AbiTarget,
            small: u8,
            signed: i16,
            wide: usize,
            float: f32,
            pointer: *const u8,
        ) -> usize {
            small as usize
                + signed.unsigned_abs() as usize
                + wide
                + float as usize
                + pointer as usize
        }
        let mixed_handler = |this: &AbiTarget,
                             small: u8,
                             signed: i16,
                             wide: usize,
                             float: f32,
                             pointer: *const u8| {
            this.observed.store(
                small as usize
                    + signed.unsigned_abs() as usize
                    + wide
                    + float as usize
                    + pointer as usize,
                Ordering::Relaxed,
            );
        };
        let mixed_context: Box<dyn DispatchControl> = unsafe {
            <_ as ThunkablePreHook<
                usize,
                AbiTarget,
                (u8, i16, usize, f32, *const u8),
            >>::into_dispatch(
                mixed_handler,
                mixed_original as *const (),
                HookId(113),
            )
        };
        let mixed_trampoline = TrampolineStorage::global()
            .unwrap()
            .make_trampoline(&*mixed_context, HookId(113))
            .unwrap();
        type MixedFunction =
            unsafe extern "C" fn(*mut AbiTarget, u8, i16, usize, f32, *const u8) -> usize;
        let mixed_function: MixedFunction = unsafe { std::mem::transmute(mixed_trampoline) };
        let pointer = ptr::without_provenance(0x9000);
        let expected = 7 + 13 + 29 + 3 + pointer as usize;

        assert_eq!(
            unsafe { mixed_function(&mut target, 7, -13, 29, 3.5, pointer) },
            expected
        );
        assert_eq!(target.observed.load(Ordering::Relaxed), expected);
    }
}
