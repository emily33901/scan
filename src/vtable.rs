//! Bounded discovery of C++ virtual-function tables.
//!
//! Discovery copies the object vptr, ABI prefix, and every function entry into
//! owned storage. It deliberately retains the null-terminator heuristic used by
//! the hook layer, but constrains it to one readable mapping and a hard cap.

use std::mem::size_of;

use thiserror::Error;

use crate::function::{ExecutableAddress, ExecutablePolicy, FunctionError};
use crate::memory::{
    Access, Address, DirectMemoryAccess, DirectMemoryAccessExt, DirectMemoryScope,
    FaultContainedMemoryAccess, FaultContainedMemoryAccessExt, MemoryError, MemoryRegion,
};

/// Absolute ceiling for null-terminated vtable discovery.
pub const HARD_MAX_VTABLE_SLOTS: usize = 65_536;

/// Supported 64-bit C++ ABI address-point layouts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CxxAbi {
    /// Microsoft x64: one complete-object-locator word precedes the table.
    Msvc64,
    /// Itanium x64: offset-to-top and type-info words precede the table.
    Itanium64,
}

impl CxxAbi {
    /// Number of pointer-sized words before the vtable address point.
    pub const fn prefix_words(self) -> usize {
        match self {
            Self::Msvc64 => 1,
            Self::Itanium64 => 2,
        }
    }

    /// ABI used by supported native hook targets.
    pub const fn native() -> Option<Self> {
        #[cfg(all(target_os = "windows", target_arch = "x86_64", target_env = "msvc"))]
        {
            Some(Self::Msvc64)
        }
        #[cfg(all(
            not(all(target_os = "windows", target_arch = "x86_64", target_env = "msvc")),
            target_os = "macos",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        {
            Some(Self::Itanium64)
        }
        #[cfg(not(any(
            all(target_os = "windows", target_arch = "x86_64", target_env = "msvc"),
            all(
                target_os = "macos",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        )))]
        {
            None
        }
    }
}

/// Limits and validation policy for one discovery operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VTableSpec {
    pub abi: CxxAbi,
    pub slot: usize,
    pub max_slots: usize,
    pub executable_policy: ExecutablePolicy,
}

impl VTableSpec {
    pub const fn new(
        abi: CxxAbi,
        slot: usize,
        max_slots: usize,
        executable_policy: ExecutablePolicy,
    ) -> Self {
        Self {
            abi,
            slot,
            max_slots,
            executable_policy,
        }
    }
}

/// Owned result of a vtable discovery operation.
#[derive(Clone, Debug)]
pub struct DiscoveredVTable {
    object: Address,
    address_point: Address,
    abi: CxxAbi,
    prefix: Box<[usize]>,
    entries: Box<[ExecutableAddress]>,
    requested_slot: usize,
}

impl DiscoveredVTable {
    pub const fn object(&self) -> Address {
        self.object
    }

    pub const fn address_point(&self) -> Address {
        self.address_point
    }

    pub const fn abi(&self) -> CxxAbi {
        self.abi
    }

    /// Copied ABI words in address order, from the lowest address to `-1`.
    pub fn prefix(&self) -> &[usize] {
        &self.prefix
    }

    /// Copied and validated non-null function entries.
    pub fn entries(&self) -> &[ExecutableAddress] {
        &self.entries
    }

    pub const fn requested_slot(&self) -> usize {
        self.requested_slot
    }

    pub fn requested_entry(&self) -> &ExecutableAddress {
        // Construction checks the requested slot against the owned entry count.
        &self.entries[self.requested_slot]
    }
}

/// Identifies which copied vtable component failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VTableRead {
    ObjectVptr,
    Prefix { index: usize },
    Entry { slot: usize },
}

/// Failures from bounded vtable discovery.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VTableError {
    #[error("vtable discovery requires 64-bit pointers, found {bytes}-byte pointers")]
    UnsupportedPointerWidth { bytes: usize },
    #[error("max_slots must be between 1 and {hard_limit}, got {requested}")]
    InvalidSlotLimit { requested: usize, hard_limit: usize },
    #[error("object address is null")]
    NullObject,
    #[error("copied object vptr is null")]
    NullVptr,
    #[error("object address {address} is not pointer-aligned")]
    MisalignedObject { address: Address },
    #[error("vtable address point {address} is not pointer-aligned")]
    MisalignedVptr { address: Address },
    #[error("address arithmetic overflow while locating {read:?}")]
    AddressOverflow { read: VTableRead },
    #[error("failed to copy {read:?}")]
    Read {
        read: VTableRead,
        #[source]
        source: MemoryError,
    },
    #[error("could not reserve owned storage for {requested} vtable words")]
    Allocation { requested: usize },
    #[error("vtable mapping boundary was reached at slot {slot} before a null terminator")]
    TerminatorMappingBoundary { slot: usize },
    #[error("vtable has no null terminator within its {max_slots}-slot limit")]
    TerminatorLimit { max_slots: usize },
    #[error("vtable entry {slot} at {address} failed executable validation")]
    InvalidExecutable {
        slot: usize,
        address: Address,
        #[source]
        source: FunctionError,
    },
    #[error("requested vtable slot {slot} is outside discovered length {len}")]
    SlotOutOfRange { slot: usize, len: usize },
}

trait VTableReader {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError>;
    fn read_word(&self, address: Address, read: VTableRead) -> Result<usize, VTableError>;
    fn executable(
        &self,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<ExecutableAddress, FunctionError>;
}

struct DirectReader<'a, M: ?Sized>(&'a M);

impl<M: DirectMemoryAccess + ?Sized> VTableReader for DirectReader<'_, M> {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        self.0.region_containing(address)
    }

    fn read_word(&self, address: Address, read: VTableRead) -> Result<usize, VTableError> {
        // SAFETY: constructing this reader is confined to `discover_direct`,
        // whose caller supplies the direct-copy lifetime and validity contract.
        unsafe { self.0.read_value(address) }.map_err(|source| VTableError::Read { read, source })
    }

    fn executable(
        &self,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<ExecutableAddress, FunctionError> {
        // SAFETY: the enclosing direct discovery has the same stability
        // contract as direct executable-address validation.
        unsafe { ExecutableAddress::from_direct(self.0, address, policy) }
    }
}

struct ScopedReader<'a, 'scope, M: ?Sized>(&'a DirectMemoryScope<'scope, M>);

impl<M: DirectMemoryAccess + ?Sized> VTableReader for ScopedReader<'_, '_, M> {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        self.0.memory().region_containing(address)
    }

    fn read_word(&self, address: Address, read: VTableRead) -> Result<usize, VTableError> {
        self.0
            .read_value(address)
            .map_err(|source| VTableError::Read { read, source })
    }

    fn executable(
        &self,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<ExecutableAddress, FunctionError> {
        ExecutableAddress::from_scope(self.0, address, policy)
    }
}

struct FaultContainedReader<'a, M: ?Sized>(&'a M);

impl<M: FaultContainedMemoryAccess + ?Sized> VTableReader for FaultContainedReader<'_, M> {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        self.0.region_containing(address)
    }

    fn read_word(&self, address: Address, read: VTableRead) -> Result<usize, VTableError> {
        self.0
            .try_read_value(address)
            .map_err(|source| VTableError::Read { read, source })
    }

    fn executable(
        &self,
        address: Address,
        policy: ExecutablePolicy,
    ) -> Result<ExecutableAddress, FunctionError> {
        ExecutableAddress::from_fault_contained(self.0, address, policy)
    }
}

/// Discovers a vtable through the direct-copy tier.
///
/// # Safety
///
/// `object`, its vptr, the ABI prefix, and every examined vtable word must stay
/// allocated, readable, initialized, and synchronized until discovery returns.
pub unsafe fn discover_direct<M: DirectMemoryAccess + ?Sized>(
    memory: &M,
    object: Address,
    spec: VTableSpec,
) -> Result<DiscoveredVTable, VTableError> {
    discover(&DirectReader(memory), object, spec)
}

/// Discovers a vtable through an already established direct-memory scope.
pub fn discover_scoped<M: DirectMemoryAccess + ?Sized>(
    scope: &DirectMemoryScope<'_, M>,
    object: Address,
    spec: VTableSpec,
) -> Result<DiscoveredVTable, VTableError> {
    discover(&ScopedReader(scope), object, spec)
}

/// Discovers a vtable through a fault-contained copying backend.
pub fn discover_fault_contained<M: FaultContainedMemoryAccess + ?Sized>(
    memory: &M,
    object: Address,
    spec: VTableSpec,
) -> Result<DiscoveredVTable, VTableError> {
    discover(&FaultContainedReader(memory), object, spec)
}

impl<M: DirectMemoryAccess + ?Sized> DirectMemoryScope<'_, M> {
    /// Discovers and copies a bounded vtable through this scope.
    pub fn discover_vtable(
        &self,
        object: Address,
        spec: VTableSpec,
    ) -> Result<DiscoveredVTable, VTableError> {
        discover_scoped(self, object, spec)
    }
}

fn discover<R: VTableReader + ?Sized>(
    reader: &R,
    object: Address,
    spec: VTableSpec,
) -> Result<DiscoveredVTable, VTableError> {
    let pointer_size = size_of::<usize>();
    if pointer_size != 8 {
        return Err(VTableError::UnsupportedPointerWidth {
            bytes: pointer_size,
        });
    }
    if spec.max_slots == 0 || spec.max_slots > HARD_MAX_VTABLE_SLOTS {
        return Err(VTableError::InvalidSlotLimit {
            requested: spec.max_slots,
            hard_limit: HARD_MAX_VTABLE_SLOTS,
        });
    }
    if object.is_null() {
        return Err(VTableError::NullObject);
    }
    if !object
        .is_aligned_to(pointer_size)
        .map_err(|source| VTableError::Read {
            read: VTableRead::ObjectVptr,
            source,
        })?
    {
        return Err(VTableError::MisalignedObject { address: object });
    }

    let address_point = Address::new(reader.read_word(object, VTableRead::ObjectVptr)?);
    if address_point.is_null() {
        return Err(VTableError::NullVptr);
    }
    if !address_point
        .is_aligned_to(pointer_size)
        .map_err(|source| VTableError::Read {
            read: VTableRead::Entry { slot: 0 },
            source,
        })?
    {
        return Err(VTableError::MisalignedVptr {
            address: address_point,
        });
    }

    let prefix_words = spec.abi.prefix_words();
    let total_words = prefix_words
        .checked_add(spec.max_slots)
        .ok_or(VTableError::Allocation {
            requested: spec.max_slots,
        })?;
    let mut prefix = Vec::new();
    prefix
        .try_reserve_exact(prefix_words)
        .map_err(|_| VTableError::Allocation {
            requested: total_words,
        })?;
    for index in 0..prefix_words {
        let distance = prefix_words - index;
        let offset = distance
            .checked_mul(pointer_size)
            .ok_or(VTableError::AddressOverflow {
                read: VTableRead::Prefix { index },
            })?;
        let address =
            address_point
                .checked_sub(offset)
                .map_err(|_| VTableError::AddressOverflow {
                    read: VTableRead::Prefix { index },
                })?;
        prefix.push(reader.read_word(address, VTableRead::Prefix { index })?);
    }

    let table_region =
        reader
            .region_containing(address_point)
            .map_err(|source| VTableError::Read {
                read: VTableRead::Entry { slot: 0 },
                source,
            })?;
    if !table_region.access.contains(Access::READ) {
        return Err(VTableError::Read {
            read: VTableRead::Entry { slot: 0 },
            source: MemoryError::PermissionDenied {
                address: address_point,
                required: Access::READ,
                actual: table_region.access,
            },
        });
    }

    let mut entries = Vec::new();
    entries
        .try_reserve_exact(spec.max_slots)
        .map_err(|_| VTableError::Allocation {
            requested: total_words,
        })?;
    let mut terminated = false;
    for slot in 0..spec.max_slots {
        let offset = slot
            .checked_mul(pointer_size)
            .ok_or(VTableError::AddressOverflow {
                read: VTableRead::Entry { slot },
            })?;
        let address =
            address_point
                .checked_add(offset)
                .map_err(|_| VTableError::AddressOverflow {
                    read: VTableRead::Entry { slot },
                })?;
        let word_range =
            address
                .checked_range(pointer_size)
                .map_err(|source| VTableError::Read {
                    read: VTableRead::Entry { slot },
                    source,
                })?;
        if !table_region.range.contains_range(word_range) {
            return Err(VTableError::TerminatorMappingBoundary { slot });
        }

        let entry = Address::new(reader.read_word(address, VTableRead::Entry { slot })?);
        if entry.is_null() {
            terminated = true;
            break;
        }
        let executable = reader
            .executable(entry, spec.executable_policy)
            .map_err(|source| VTableError::InvalidExecutable {
                slot,
                address: entry,
                source,
            })?;
        entries.push(executable);
    }

    if !terminated {
        return Err(VTableError::TerminatorLimit {
            max_slots: spec.max_slots,
        });
    }
    if spec.slot >= entries.len() {
        return Err(VTableError::SlotOutOfRange {
            slot: spec.slot,
            len: entries.len(),
        });
    }

    Ok(DiscoveredVTable {
        object,
        address_point,
        abi: spec.abi,
        prefix: prefix.into_boxed_slice(),
        entries: entries.into_boxed_slice(),
        requested_slot: spec.slot,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::memory::{AddressRange, MemoryMap, ModuleId, RegionKind, DEFAULT_MAX_REGIONS};

    const NEVER_FAIL: usize = usize::MAX;

    struct FakeMemory {
        regions: Vec<MemoryRegion>,
        reads: AtomicUsize,
        fail_read_at: usize,
    }

    impl FakeMemory {
        fn new(regions: Vec<MemoryRegion>) -> Self {
            Self {
                regions,
                reads: AtomicUsize::new(0),
                fail_read_at: NEVER_FAIL,
            }
        }

        fn fail_read_at(mut self, read: usize) -> Self {
            self.fail_read_at = read;
            self
        }

        unsafe fn copy(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
            let read = self.reads.fetch_add(1, Ordering::Relaxed);
            if read == self.fail_read_at {
                return Err(MemoryError::PartialRead {
                    expected: output.len(),
                    actual: 0,
                });
            }
            let range = address.checked_range(output.len())?;
            self.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)?;
            // SAFETY: test regions are built only over the fixture's live Box
            // allocations, and check_range rejects every other source.
            unsafe {
                std::ptr::copy(
                    address.get() as *const u8,
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            Ok(())
        }
    }

    impl MemoryMap for FakeMemory {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            self.regions
                .iter()
                .copied()
                .find(|region| region.contains(address))
                .ok_or(MemoryError::Unmapped { address })
        }
    }

    impl DirectMemoryAccess for FakeMemory {
        unsafe fn read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            // SAFETY: delegated to this test backend's caller.
            unsafe { self.copy(address, output) }
        }

        unsafe fn write_bytes_from(
            &self,
            _address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            Err(MemoryError::PermissionDenied {
                address: _address,
                required: Access::WRITE,
                actual: Access::READ,
            })
        }
    }

    unsafe impl FaultContainedMemoryAccess for FakeMemory {
        fn try_read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            // SAFETY: copy first validates that the complete source lies in a
            // fixture-owned allocation, so arbitrary addresses fail safely.
            unsafe { self.copy(address, output) }
        }

        unsafe fn try_write_bytes_from(
            &self,
            address: Address,
            _bytes: &[u8],
        ) -> Result<(), MemoryError> {
            Err(MemoryError::PermissionDenied {
                address,
                required: Access::WRITE,
                actual: Access::READ,
            })
        }
    }

    fn first_function() {}
    fn second_function() {}

    fn function_address(function: fn()) -> usize {
        function as *const () as usize
    }

    struct Fixture {
        _layout: Box<[usize]>,
        _object: Box<usize>,
        memory: FakeMemory,
        object: Address,
        address_point: Address,
    }

    impl Fixture {
        fn new(abi: CxxAbi, entries: &[usize], terminator: bool, module: ModuleId) -> Self {
            let prefix: &[usize] = match abi {
                CxxAbi::Msvc64 => &[0xaaaa],
                CxxAbi::Itanium64 => &[usize::MAX, 0xbbbb],
            };
            let mut words =
                Vec::with_capacity(prefix.len() + entries.len() + usize::from(terminator));
            words.extend_from_slice(prefix);
            words.extend_from_slice(entries);
            if terminator {
                words.push(0);
            }
            let layout = words.into_boxed_slice();
            let address_point = Address::new(
                // SAFETY: layout contains at least the ABI prefix, and the
                // address point may legally be one-past-prefix before entries.
                unsafe { layout.as_ptr().add(prefix.len()) } as usize,
            );
            let object = Box::new(address_point.get());
            let object_address = Address::new((&*object as *const usize) as usize);

            let mut regions = vec![
                region(
                    Address::new(layout.as_ptr() as usize),
                    layout.len() * size_of::<usize>(),
                    Access::READ,
                    RegionKind::Private,
                    None,
                ),
                region(
                    object_address,
                    size_of::<usize>(),
                    Access::READ,
                    RegionKind::Private,
                    None,
                ),
            ];
            for entry in entries.iter().copied() {
                regions.push(region(
                    Address::new(entry),
                    1,
                    Access::READ | Access::EXECUTE,
                    RegionKind::Image,
                    Some(module),
                ));
            }

            Self {
                _layout: layout,
                _object: object,
                memory: FakeMemory::new(regions),
                object: object_address,
                address_point,
            }
        }

        fn spec(&self, abi: CxxAbi, slot: usize, max_slots: usize) -> VTableSpec {
            VTableSpec::new(abi, slot, max_slots, ExecutablePolicy::AnyExecutable)
        }
    }

    fn region(
        start: Address,
        len: usize,
        access: Access,
        kind: RegionKind,
        module: Option<ModuleId>,
    ) -> MemoryRegion {
        MemoryRegion {
            range: AddressRange::new(start, len).unwrap(),
            access,
            kind,
            module,
        }
    }

    #[test]
    fn copies_msvc_prefix_entries_and_terminator() {
        let functions = [
            function_address(first_function),
            function_address(second_function),
        ];
        let fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        let found = unsafe {
            discover_direct(
                &fixture.memory,
                fixture.object,
                fixture.spec(CxxAbi::Msvc64, 1, 8),
            )
            .unwrap()
        };

        assert_eq!(found.address_point(), fixture.address_point);
        assert_eq!(found.prefix(), &[0xaaaa]);
        assert_eq!(found.entries().len(), 2);
        assert_eq!(
            found.requested_entry().address(),
            Address::new(functions[1])
        );
    }

    #[test]
    fn scoped_discovery_copies_itanium_prefix() {
        let functions = [function_address(first_function)];
        let fixture = Fixture::new(CxxAbi::Itanium64, &functions, true, ModuleId(7));
        let found = unsafe {
            crate::memory::with_direct_memory_scope(&fixture.memory, |scope| {
                scope
                    .discover_vtable(fixture.object, fixture.spec(CxxAbi::Itanium64, 0, 4))
                    .unwrap()
            })
        };
        assert_eq!(found.prefix(), &[usize::MAX, 0xbbbb]);
    }

    #[test]
    fn fault_contained_discovery_uses_the_same_algorithm() {
        let functions = [function_address(first_function)];
        let fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        let found = discover_fault_contained(
            &fixture.memory,
            fixture.object,
            fixture.spec(CxxAbi::Msvc64, 0, 4),
        )
        .unwrap();
        assert_eq!(found.entries().len(), 1);
    }

    #[test]
    fn rejects_a_boundary_before_the_terminator() {
        let functions = [function_address(first_function)];
        let mut fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        fixture.memory.regions[0].range = AddressRange::new(
            Address::new(fixture._layout.as_ptr() as usize),
            2 * size_of::<usize>(),
        )
        .unwrap();

        let error = unsafe {
            discover_direct(
                &fixture.memory,
                fixture.object,
                fixture.spec(CxxAbi::Msvc64, 0, 4),
            )
            .unwrap_err()
        };
        assert!(matches!(
            error,
            VTableError::TerminatorMappingBoundary { slot: 1 }
        ));
    }

    #[test]
    fn rejects_a_cap_before_the_terminator() {
        let functions = [function_address(first_function)];
        let fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        let error = unsafe {
            discover_direct(
                &fixture.memory,
                fixture.object,
                fixture.spec(CxxAbi::Msvc64, 0, 1),
            )
            .unwrap_err()
        };
        assert!(matches!(
            error,
            VTableError::TerminatorLimit { max_slots: 1 }
        ));
    }

    #[test]
    fn requested_slot_must_have_been_discovered() {
        let functions = [function_address(first_function)];
        let fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        let error = unsafe {
            discover_direct(
                &fixture.memory,
                fixture.object,
                fixture.spec(CxxAbi::Msvc64, 1, 4),
            )
            .unwrap_err()
        };
        assert!(matches!(
            error,
            VTableError::SlotOutOfRange { slot: 1, len: 1 }
        ));
    }

    #[test]
    fn executable_module_policy_is_enforced() {
        let functions = [function_address(first_function)];
        let fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        let spec = VTableSpec::new(CxxAbi::Msvc64, 0, 4, ExecutablePolicy::Module(ModuleId(8)));
        let error = unsafe { discover_direct(&fixture.memory, fixture.object, spec).unwrap_err() };
        assert!(matches!(
            error,
            VTableError::InvalidExecutable { slot: 0, .. }
        ));
    }

    #[test]
    fn propagates_a_partial_entry_read_with_context() {
        let functions = [function_address(first_function)];
        let mut fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        fixture.memory = fixture.memory.fail_read_at(2);
        let error = unsafe {
            discover_direct(
                &fixture.memory,
                fixture.object,
                fixture.spec(CxxAbi::Msvc64, 0, 4),
            )
            .unwrap_err()
        };
        assert!(matches!(
            error,
            VTableError::Read {
                read: VTableRead::Entry { slot: 0 },
                source: MemoryError::PartialRead { .. }
            }
        ));
    }

    #[test]
    fn validates_the_library_hard_cap() {
        let functions = [function_address(first_function)];
        let fixture = Fixture::new(CxxAbi::Msvc64, &functions, true, ModuleId(7));
        for max_slots in [0, HARD_MAX_VTABLE_SLOTS + 1] {
            let error = unsafe {
                discover_direct(
                    &fixture.memory,
                    fixture.object,
                    fixture.spec(CxxAbi::Msvc64, 0, max_slots),
                )
                .unwrap_err()
            };
            assert!(matches!(error, VTableError::InvalidSlotLimit { .. }));
        }
    }
}
