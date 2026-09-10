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
    FaultContainedMemoryAccess, MemoryError, MemoryMap, ModuleId,
};
use crate::module::{LoadedModule, Section};

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
    /// Require an address proven to be in a selected image snapshot.
    Image(ModuleAddressPolicy),
    /// Require an address proven to be in a selected module snapshot.
    Module(ModuleAddressPolicy),
    /// Require membership in a section proven to belong to parsed module
    /// metadata. The opaque proof cannot be assembled from an arbitrary range.
    ExecutableSection(ExecutableSectionPolicy),
}

impl ExecutablePolicy {
    /// Builds an image-membership policy for one address from copied, parsed
    /// module metadata.
    pub fn for_image(module: &LoadedModule, address: Address) -> Result<Self, FunctionError> {
        Ok(Self::Image(ModuleAddressPolicy::new(module, address)?))
    }

    /// Builds a module-membership policy for one address from copied, parsed
    /// module metadata.
    pub fn for_module(module: &LoadedModule, address: Address) -> Result<Self, FunctionError> {
        Ok(Self::Module(ModuleAddressPolicy::new(module, address)?))
    }

    /// Builds a policy from an executable section in parsed module metadata.
    ///
    /// `section` must be an element of `module.sections`, not merely a caller-
    /// supplied value with equal fields. The returned policy owns only copied
    /// snapshot identity and must still be revalidated at use time.
    pub fn for_executable_section(
        module: &LoadedModule,
        section: &Section,
        address: Address,
    ) -> Result<Self, FunctionError> {
        let belongs_to_module = module
            .sections
            .iter()
            .any(|candidate| std::ptr::eq(candidate, section));
        if !belongs_to_module || !module.contains_range(section.range) {
            return Err(FunctionError::SectionNotInModule {
                module: module.id,
                range: section.range,
            });
        }
        if !section.is_executable() {
            return Err(FunctionError::SectionNotExecutable {
                module: module.id,
                range: section.range,
            });
        }
        if !section.range.contains(address) {
            return Err(FunctionError::AddressOutsideSection {
                address,
                module: module.id,
                range: section.range,
            });
        }
        Ok(Self::ExecutableSection(ExecutableSectionPolicy {
            module: module.id,
            range: section.range,
            address,
        }))
    }
}

/// Opaque proof that one address belonged to a copied module snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModuleAddressPolicy {
    module: ModuleId,
    address: Address,
}

impl ModuleAddressPolicy {
    fn new(module: &LoadedModule, address: Address) -> Result<Self, FunctionError> {
        if !module.contains_address(address) {
            return Err(FunctionError::AddressOutsideModule {
                address,
                module: module.id,
            });
        }
        Ok(Self {
            module: module.id,
            address,
        })
    }

    pub const fn module(self) -> ModuleId {
        self.module
    }

    pub const fn address(self) -> Address {
        self.address
    }
}

/// Opaque proof that a range came from an executable section in parsed module
/// metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutableSectionPolicy {
    module: ModuleId,
    range: AddressRange,
    address: Address,
}

impl ExecutableSectionPolicy {
    pub const fn module(self) -> ModuleId {
        self.module
    }

    pub const fn range(self) -> AddressRange {
        self.range
    }

    pub const fn address(self) -> Address {
        self.address
    }
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
        ExecutablePolicy::Image(proof) | ExecutablePolicy::Module(proof)
            if proof.address == address =>
        {
            Ok(())
        }
        ExecutablePolicy::Image(proof) | ExecutablePolicy::Module(proof) => {
            Err(FunctionError::PolicyAddressMismatch {
                address,
                authorized: proof.address,
                module: proof.module,
            })
        }
        ExecutablePolicy::ExecutableSection(section)
            if section.address == address && section.range.contains(address) =>
        {
            Ok(())
        }
        ExecutablePolicy::ExecutableSection(section) => {
            Err(FunctionError::ExecutableSectionMismatch {
                address,
                expected_module: section.module,
                range: section.range,
                authorized: section.address,
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

    /// Returns the untyped code address without making an extra-indirection or
    /// callable-signature claim.
    pub fn as_code_ptr(self) -> *const () {
        self.address().get() as *const ()
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
    #[error("address {address} was not in copied module snapshot {module:?}")]
    AddressOutsideModule { address: Address, module: ModuleId },
    #[error("address {address} is outside section {range:?} of module {module:?}")]
    AddressOutsideSection {
        address: Address,
        module: ModuleId,
        range: AddressRange,
    },
    #[error(
        "policy for module {module:?} authorizes address {authorized}, not requested address {address}"
    )]
    PolicyAddressMismatch {
        address: Address,
        authorized: Address,
        module: ModuleId,
    },
    #[error(
        "executable-section policy for module {expected_module:?} authorizes {authorized} in {range:?}, not requested address {address}"
    )]
    ExecutableSectionMismatch {
        address: Address,
        expected_module: ModuleId,
        range: AddressRange,
        authorized: Address,
    },
    #[error("range {range:?} is not the selected section of module {module:?}")]
    SectionNotInModule {
        module: ModuleId,
        range: AddressRange,
    },
    #[error("section {range:?} of module {module:?} is not executable")]
    SectionNotExecutable {
        module: ModuleId,
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
    use crate::memory::{MemoryRegion, PlatformOperation, RegionKind};
    use crate::module::{ImageFormat, SectionKind};
    use std::path::PathBuf;

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

    fn parsed_module(module: ModuleId) -> LoadedModule {
        let image = AddressRange::new(Address::new(0x1_0000), 0x1000).unwrap();
        LoadedModule {
            id: module,
            path: PathBuf::from("fixture"),
            image,
            mapped_ranges: vec![image],
            format: ImageFormat::Pe,
            sections: vec![Section {
                name: ".text".to_owned(),
                range: AddressRange::new(Address::new(0x1_0080), 0x100).unwrap(),
                access: Access::READ | Access::EXECUTE,
                kind: SectionKind::Code,
            }],
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
        let parsed = parsed_module(module);
        let address = Address::new(0x1_0100);
        let image_policy = ExecutablePolicy::for_image(&parsed, address).unwrap();
        let module_policy = ExecutablePolicy::for_module(&parsed, address).unwrap();
        let section_policy =
            ExecutablePolicy::for_executable_section(&parsed, &parsed.sections[0], address)
                .unwrap();
        // Production mapping backends do not need to synthesize module IDs or
        // classify mappings as images for copied module proof to remain useful.
        let memory = memory(Access::EXECUTE, RegionKind::Shared, None);
        assert!(ExecutableAddress::from_fault_contained(&memory, address, image_policy).is_ok());
        assert!(ExecutableAddress::from_fault_contained(&memory, address, module_policy).is_ok());
        assert!(ExecutableAddress::from_fault_contained(&memory, address, section_policy).is_ok());

        let wrong_address = Address::new(0x1_0110);
        let wrong = ExecutableAddress::from_fault_contained(&memory, wrong_address, module_policy);
        assert!(matches!(
            wrong,
            Err(FunctionError::PolicyAddressMismatch { .. })
        ));
    }

    #[test]
    fn arbitrary_private_range_cannot_be_mislabeled_as_a_module_section() {
        let module = ModuleId(7);
        let parsed = parsed_module(module);
        let copied_section = parsed.sections[0].clone();
        assert!(matches!(
            ExecutablePolicy::for_executable_section(
                &parsed,
                &copied_section,
                Address::new(0x1_0100)
            ),
            Err(FunctionError::SectionNotInModule { .. })
        ));

        let jit_address = Address::new(0x2_0100);
        assert!(matches!(
            ExecutablePolicy::for_module(&parsed, jit_address),
            Err(FunctionError::AddressOutsideModule { .. })
        ));

        let section_address = Address::new(0x1_0100);
        let policy =
            ExecutablePolicy::for_executable_section(&parsed, &parsed.sections[0], section_address)
                .unwrap();
        let production_shaped = memory(Access::EXECUTE, RegionKind::Private, None);
        assert!(ExecutableAddress::from_fault_contained(
            &production_shaped,
            section_address,
            policy
        )
        .is_ok());
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
        assert_eq!(function.as_code_ptr().addr(), 0x1_0100);
    }
}
