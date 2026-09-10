//! Checked executable addresses and typed callable intent.
//!
//! These wrappers validate the mapping that exists at the time of each check.
//! They do not pin code, select an ABI, or prove that a module will remain
//! loaded. Callers must therefore validate again immediately before converting
//! an address to a callable value.

use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};

use thiserror::Error;

use crate::memory::{
    Access, Address, AddressRange, DirectMemoryAccess, DirectMemoryScope,
    FaultContainedMemoryAccess, MemoryError, MemoryMap, ModuleId, RegionKind,
};

/// Alignment required for an instruction address on the current target.
///
/// AArch64 instructions are four-byte aligned. x86 and x86-64 permit an
/// instruction to begin at any byte address.
#[cfg(target_arch = "aarch64")]
pub const EXECUTABLE_ADDRESS_ALIGNMENT: usize = 4;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const EXECUTABLE_ADDRESS_ALIGNMENT: usize = 1;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
pub const EXECUTABLE_ADDRESS_ALIGNMENT: usize = align_of::<usize>();

/// Additional membership required of an executable address.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutablePolicy {
    /// Permit any currently executable mapping, including private JIT memory.
    #[default]
    AnyExecutable,
    /// Require an executable mapping classified as an image by the platform.
    Image,
    /// Require the currently mapped region to belong to this module snapshot.
    Module(ModuleId),
    /// Require a selected module and a validated executable-section range.
    ExecutableSection {
        module: ModuleId,
        range: AddressRange,
    },
}

/// An address that satisfied an executable policy when last validated.
///
/// This value is a copied address, not a lease on its mapping. Use one of the
/// `revalidate_*` methods immediately before converting or calling it.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ExecutableAddress {
    address: Address,
    policy: ExecutablePolicy,
}

impl ExecutableAddress {
    /// Validates an address through the raw direct tier.
    ///
    /// # Safety
    ///
    /// The caller must uphold the direct backend's mapping-stability contract
    /// through this validation and any immediate use based on its result.
    pub unsafe fn from_direct<M: DirectMemoryAccess + ?Sized>(
        memory: &M,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<Self, FunctionError> {
        validate(memory, address, policy)?;
        Ok(Self { address, policy })
    }

    /// Validates an address under an established direct-memory scope.
    pub fn from_scope<M: DirectMemoryAccess + ?Sized>(
        scope: &DirectMemoryScope<'_, M>,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<Self, FunctionError> {
        validate(scope.memory(), address, policy)?;
        Ok(Self { address, policy })
    }

    /// Validates an address through a fault-contained backend.
    pub fn from_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        memory: &M,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<Self, FunctionError> {
        validate(memory, address, policy)?;
        Ok(Self { address, policy })
    }

    pub const fn address(self) -> Address {
        self.address
    }

    pub const fn policy(self) -> ExecutablePolicy {
        self.policy
    }

    /// Rechecks current permissions and policy through the raw direct tier.
    ///
    /// # Safety
    ///
    /// The caller must uphold the direct backend's mapping-stability contract
    /// through this validation and any immediate use based on its result.
    pub unsafe fn revalidate_direct<M: DirectMemoryAccess + ?Sized>(
        &self,
        memory: &M,
    ) -> Result<(), FunctionError> {
        validate(memory, self.address, self.policy)
    }

    /// Rechecks current permissions and policy under a direct-memory scope.
    pub fn revalidate_scope<M: DirectMemoryAccess + ?Sized>(
        &self,
        scope: &DirectMemoryScope<'_, M>,
    ) -> Result<(), FunctionError> {
        validate(scope.memory(), self.address, self.policy)
    }

    /// Rechecks current permissions and policy through a fault-contained tier.
    pub fn revalidate_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
        &self,
        memory: &M,
    ) -> Result<(), FunctionError> {
        validate(memory, self.address, self.policy)
    }
}

impl fmt::Debug for ExecutableAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutableAddress")
            .field("address", &self.address)
            .field("policy", &self.policy)
            .finish()
    }
}

fn validate(
    memory: &(impl MemoryMap + ?Sized),
    address: Address,
    policy: ExecutablePolicy,
) -> Result<(), FunctionError> {
    address
        .require_non_null()
        .map_err(|source| FunctionError::Memory { source })?;
    address
        .require_alignment(EXECUTABLE_ADDRESS_ALIGNMENT)
        .map_err(|source| FunctionError::Memory { source })?;

    let region = memory
        .region_containing(address)
        .map_err(|source| FunctionError::Memory { source })?;
    if !region.contains(address) || region.range.end() <= address {
        return Err(FunctionError::Memory {
            source: MemoryError::NonProgressingRegion { address },
        });
    }
    if !region.access.contains(Access::EXECUTE) {
        return Err(FunctionError::Memory {
            source: MemoryError::PermissionDenied {
                address,
                required: Access::EXECUTE,
                actual: region.access,
            },
        });
    }

    match policy {
        ExecutablePolicy::AnyExecutable => Ok(()),
        ExecutablePolicy::Image if region.kind == RegionKind::Image => Ok(()),
        ExecutablePolicy::Image => Err(FunctionError::ImageRequired { address }),
        ExecutablePolicy::Module(expected) if region.module == Some(expected) => Ok(()),
        ExecutablePolicy::Module(expected) => Err(FunctionError::ModuleMismatch {
            address,
            expected,
            actual: region.module,
        }),
        ExecutablePolicy::ExecutableSection { module, range }
            if region.module == Some(module) && range.contains(address) =>
        {
            Ok(())
        }
        ExecutablePolicy::ExecutableSection { module, range } => {
            Err(FunctionError::ExecutableSectionMismatch {
                address,
                expected_module: module,
                actual_module: region.module,
                range,
            })
        }
    }
}

/// An executable address tagged with the intended callable signature.
///
/// `Signature` is documentary until the unsafe conversion boundary. Rust has
/// no stable trait implemented only by function-pointer types, so callers must
/// choose a function pointer with the correct ABI and signature.
pub struct FunctionAddress<Signature> {
    executable: ExecutableAddress,
    _signature: PhantomData<fn() -> Signature>,
}

impl<Signature> FunctionAddress<Signature> {
    pub const fn new(executable: ExecutableAddress) -> Self {
        Self {
            executable,
            _signature: PhantomData,
        }
    }

    pub const fn executable(self) -> ExecutableAddress {
        self.executable
    }

    pub const fn address(self) -> Address {
        self.executable.address()
    }

    /// Returns a typed raw pointer after the caller establishes use-time
    /// validity.
    ///
    /// # Safety
    ///
    /// The address must have been revalidated immediately before this call and
    /// must remain executable for the use of the returned pointer. `Signature`
    /// must describe the intended function's exact ABI and signature.
    pub unsafe fn as_ptr(self) -> *const Signature {
        self.address().get() as *const Signature
    }

    /// Converts the address into a function-pointer value.
    ///
    /// # Safety
    ///
    /// `Signature` must be a function-pointer type of pointer width with the
    /// exact ABI and signature of the target. The mapping must have been
    /// revalidated immediately before conversion and remain live and executable
    /// through every call. Calling the result remains unsafe whenever the
    /// function-pointer type or the target function requires it.
    pub unsafe fn into_callable(self) -> Result<Signature, FunctionError>
    where
        Signature: Copy,
    {
        let actual = size_of::<Signature>();
        let expected = size_of::<usize>();
        if actual != expected || align_of::<Signature>() > align_of::<usize>() {
            return Err(FunctionError::InvalidSignatureLayout {
                size: actual,
                alignment: align_of::<Signature>(),
                pointer_size: expected,
                pointer_alignment: align_of::<usize>(),
            });
        }
        let address = self.address().get();
        // SAFETY: layout was checked above. The remaining validity requirements
        // are delegated to this method's caller because stable Rust cannot
        // express a function-pointer-only bound.
        Ok(unsafe { std::mem::transmute_copy::<usize, Signature>(&address) })
    }
}

impl<Signature> Copy for FunctionAddress<Signature> {}

impl<Signature> Clone for FunctionAddress<Signature> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Signature> fmt::Debug for FunctionAddress<Signature> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("FunctionAddress")
            .field(&self.executable)
            .finish()
    }
}

impl<Signature> From<ExecutableAddress> for FunctionAddress<Signature> {
    fn from(executable: ExecutableAddress) -> Self {
        Self::new(executable)
    }
}

/// A pointer-sized slot index in a virtual-function table.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct VTableSlot(usize);

impl VTableSlot {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn from_byte_offset(offset: usize) -> Result<Self, FunctionError> {
        let pointer_width = size_of::<usize>();
        if offset % pointer_width != 0 {
            return Err(FunctionError::InvalidVTableOffset {
                offset,
                pointer_width,
            });
        }
        Ok(Self(offset / pointer_width))
    }

    pub const fn index(self) -> usize {
        self.0
    }

    pub fn byte_offset(self) -> Result<usize, FunctionError> {
        self.0
            .checked_mul(size_of::<usize>())
            .ok_or(FunctionError::VTableOffsetOverflow { slot: self.0 })
    }
}

/// Failures while validating or representing callable addresses.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum FunctionError {
    #[error("executable-address memory validation failed")]
    Memory {
        #[source]
        source: MemoryError,
    },
    #[error("address {address} is executable but is not in an image mapping")]
    ImageRequired { address: Address },
    #[error("address {address} belongs to module {actual:?}, not expected module {expected:?}")]
    ModuleMismatch {
        address: Address,
        expected: ModuleId,
        actual: Option<ModuleId>,
    },
    #[error(
        "address {address} is outside executable section {range:?} of module {expected_module:?}; current module is {actual_module:?}"
    )]
    ExecutableSectionMismatch {
        address: Address,
        expected_module: ModuleId,
        actual_module: Option<ModuleId>,
        range: AddressRange,
    },
    #[error("vtable byte offset {offset} is not divisible by pointer width {pointer_width}")]
    InvalidVTableOffset { offset: usize, pointer_width: usize },
    #[error("vtable slot {slot} overflows its byte offset")]
    VTableOffsetOverflow { slot: usize },
    #[error(
        "callable signature layout ({size} bytes, alignment {alignment}) does not match pointer layout ({pointer_size} bytes, alignment {pointer_alignment})"
    )]
    InvalidSignatureLayout {
        size: usize,
        alignment: usize,
        pointer_size: usize,
        pointer_alignment: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryRegion, PlatformOperation};

    #[derive(Clone, Copy)]
    struct TestMemory {
        region: MemoryRegion,
    }

    impl MemoryMap for TestMemory {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            if self.region.contains(address) {
                Ok(self.region)
            } else {
                Err(MemoryError::Unmapped { address })
            }
        }
    }

    impl DirectMemoryAccess for TestMemory {
        unsafe fn read_bytes_into(
            &self,
            _address: Address,
            _output: &mut [u8],
        ) -> Result<(), MemoryError> {
            Err(MemoryError::UnsupportedPlatform {
                operation: PlatformOperation::Read,
            })
        }

        unsafe fn write_bytes_from(
            &self,
            _address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            Err(MemoryError::UnsupportedPlatform {
                operation: PlatformOperation::Write,
            })
        }
    }

    unsafe impl FaultContainedMemoryAccess for TestMemory {
        fn try_read_bytes_into(
            &self,
            _address: Address,
            _output: &mut [u8],
        ) -> Result<(), MemoryError> {
            Err(MemoryError::UnsupportedPlatform {
                operation: PlatformOperation::Read,
            })
        }

        unsafe fn try_write_bytes_from(
            &self,
            _address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            Err(MemoryError::UnsupportedPlatform {
                operation: PlatformOperation::Write,
            })
        }
    }

    fn memory(access: Access, kind: RegionKind, module: Option<ModuleId>) -> TestMemory {
        TestMemory {
            region: MemoryRegion {
                range: AddressRange::new(Address::new(0x1_0000), 0x1000).unwrap(),
                access,
                kind,
                module,
            },
        }
    }

    #[test]
    fn default_policy_accepts_private_executable_memory() {
        let memory = memory(Access::READ | Access::EXECUTE, RegionKind::Private, None);
        let result = ExecutableAddress::from_fault_contained(
            &memory,
            Address::new(0x1_0100),
            ExecutablePolicy::default(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn validation_rejects_null_and_non_executable_addresses() {
        let memory = memory(Access::READ, RegionKind::Private, None);
        let null = ExecutableAddress::from_fault_contained(
            &memory,
            Address::NULL,
            ExecutablePolicy::AnyExecutable,
        );
        assert!(matches!(
            null,
            Err(FunctionError::Memory {
                source: MemoryError::NullAddress
            })
        ));

        let denied = ExecutableAddress::from_fault_contained(
            &memory,
            Address::new(0x1_0100),
            ExecutablePolicy::AnyExecutable,
        );
        assert!(matches!(
            denied,
            Err(FunctionError::Memory {
                source: MemoryError::PermissionDenied { .. }
            })
        ));
    }

    #[test]
    fn stricter_policies_check_current_region_metadata() {
        let module = ModuleId(7);
        let memory = memory(Access::EXECUTE, RegionKind::Image, Some(module));
        let address = Address::new(0x1_0100);
        assert!(
            ExecutableAddress::from_fault_contained(&memory, address, ExecutablePolicy::Image)
                .is_ok()
        );
        assert!(ExecutableAddress::from_fault_contained(
            &memory,
            address,
            ExecutablePolicy::Module(module)
        )
        .is_ok());
        assert!(ExecutableAddress::from_fault_contained(
            &memory,
            address,
            ExecutablePolicy::ExecutableSection {
                module,
                range: AddressRange::new(Address::new(0x1_0080), 0x100).unwrap(),
            }
        )
        .is_ok());

        let wrong = ExecutableAddress::from_fault_contained(
            &memory,
            address,
            ExecutablePolicy::Module(ModuleId(8)),
        );
        assert!(matches!(wrong, Err(FunctionError::ModuleMismatch { .. })));
    }

    #[test]
    fn revalidation_observes_lost_execute_permission() {
        let executable = memory(Access::EXECUTE, RegionKind::Private, None);
        let address = ExecutableAddress::from_fault_contained(
            &executable,
            Address::new(0x1_0100),
            ExecutablePolicy::AnyExecutable,
        )
        .unwrap();
        let stale = memory(Access::READ, RegionKind::Private, None);
        assert!(matches!(
            address.revalidate_fault_contained(&stale),
            Err(FunctionError::Memory {
                source: MemoryError::PermissionDenied { .. }
            })
        ));
    }

    #[test]
    fn scoped_validation_uses_the_direct_tier() {
        let memory = memory(Access::EXECUTE, RegionKind::Private, None);
        // SAFETY: this test performs no copy and the fixed map remains valid for
        // the complete non-escaping operation.
        let result = unsafe {
            crate::memory::with_direct_memory_scope(&memory, |scope| {
                ExecutableAddress::from_scope(
                    scope,
                    Address::new(0x1_0100),
                    ExecutablePolicy::AnyExecutable,
                )
            })
        };
        assert!(result.is_ok());
    }

    #[test]
    fn vtable_offsets_must_be_pointer_aligned() {
        let width = size_of::<usize>();
        assert_eq!(VTableSlot::from_byte_offset(width * 3).unwrap().index(), 3);
        assert!(matches!(
            VTableSlot::from_byte_offset(width + 1),
            Err(FunctionError::InvalidVTableOffset { .. })
        ));
    }

    #[test]
    fn function_address_records_signature_without_claiming_an_abi() {
        type Callback = unsafe extern "C" fn(usize) -> usize;

        let memory = memory(Access::EXECUTE, RegionKind::Private, None);
        let executable = ExecutableAddress::from_fault_contained(
            &memory,
            Address::new(0x1_0100),
            ExecutablePolicy::AnyExecutable,
        )
        .unwrap();
        let function = FunctionAddress::<Callback>::new(executable);
        assert_eq!(function.address(), Address::new(0x1_0100));
    }
}
