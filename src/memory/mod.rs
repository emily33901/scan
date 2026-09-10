//! Checked address arithmetic and process-memory access contracts.
//!
//! [`MemoryMap`] performs bounded, complete-range validation. That validation
//! is deliberately separate from copying: mappings are snapshots and cannot
//! prove allocation lifetime, initialization, or synchronization.

use std::fmt;

use bitflags::bitflags;
use thiserror::Error;

/// An address in the current process.
///
/// Zero is representable so that copied pointer values can be inspected before
/// an operation explicitly requires them to be non-null.
#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Address(usize);

impl Address {
    pub const NULL: Self = Self(0);

    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn get(self) -> usize {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn require_non_null(self) -> Result<Self, MemoryError> {
        if self.is_null() {
            Err(MemoryError::NullAddress)
        } else {
            Ok(self)
        }
    }

    pub fn checked_add(self, offset: usize) -> Result<Self, MemoryError> {
        self.0
            .checked_add(offset)
            .map(Self)
            .ok_or(MemoryError::AddressOverflow)
    }

    pub fn checked_sub(self, offset: usize) -> Result<Self, MemoryError> {
        self.0
            .checked_sub(offset)
            .map(Self)
            .ok_or(MemoryError::AddressOverflow)
    }

    pub fn checked_add_signed(self, offset: i64) -> Result<Self, MemoryError> {
        let magnitude =
            usize::try_from(offset.unsigned_abs()).map_err(|_| MemoryError::AddressOverflow)?;
        let value = if offset >= 0 {
            self.0.checked_add(magnitude)
        } else {
            self.0.checked_sub(magnitude)
        };
        value.map(Self).ok_or(MemoryError::AddressOverflow)
    }

    pub fn checked_range(self, len: usize) -> Result<AddressRange, MemoryError> {
        AddressRange::new(self, len)
    }

    pub fn is_aligned_to(self, alignment: usize) -> Result<bool, MemoryError> {
        validate_alignment(alignment)?;
        Ok(self.0 & (alignment - 1) == 0)
    }

    pub fn require_alignment(self, alignment: usize) -> Result<Self, MemoryError> {
        if self.is_aligned_to(alignment)? {
            Ok(self)
        } else {
            Err(MemoryError::Misaligned {
                address: self,
                alignment,
            })
        }
    }
}

impl From<usize> for Address {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<Address> for usize {
    fn from(value: Address) -> Self {
        value.0
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address({self:#x})")
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

impl fmt::LowerHex for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

/// A non-empty half-open address range.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AddressRange {
    start: Address,
    end: Address,
}

impl AddressRange {
    pub fn new(start: Address, len: usize) -> Result<Self, MemoryError> {
        if len == 0 {
            return Err(MemoryError::ZeroLength);
        }
        let end = start.checked_add(len)?;
        Ok(Self { start, end })
    }

    pub fn from_bounds(start: Address, end: Address) -> Result<Self, MemoryError> {
        if end <= start {
            return if end == start {
                Err(MemoryError::ZeroLength)
            } else {
                Err(MemoryError::AddressOverflow)
            };
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> Address {
        self.start
    }

    pub const fn end(self) -> Address {
        self.end
    }

    pub const fn len(self) -> usize {
        self.end.0 - self.start.0
    }

    pub const fn contains(self, address: Address) -> bool {
        self.start.0 <= address.0 && address.0 < self.end.0
    }

    pub const fn contains_range(self, other: Self) -> bool {
        self.start.0 <= other.start.0 && other.end.0 <= self.end.0
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }
}

bitflags! {
    /// Permissions required from or reported for a memory region.
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct Access: u8 {
        const READ = 1 << 0;
        const WRITE = 1 << 1;
        const EXECUTE = 1 << 2;
    }
}

/// Platform-neutral classification of a memory mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegionKind {
    Private,
    Image,
    Mapped,
    Shared,
    Unknown,
}

/// Stable identity assigned to a discovered module snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ModuleId(pub u64);

/// A single contiguous memory mapping returned by a backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryRegion {
    pub range: AddressRange,
    pub access: Access,
    pub kind: RegionKind,
    pub module: Option<ModuleId>,
}

impl MemoryRegion {
    pub const fn contains(self, address: Address) -> bool {
        self.range.contains(address)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformOperation {
    Query,
    Read,
    Write,
}

/// Failures produced by checked address and memory operations.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum MemoryError {
    #[error("null address is not valid for this operation")]
    NullAddress,
    #[error("address arithmetic overflow or underflow")]
    AddressOverflow,
    #[error("zero-length memory operations are not supported")]
    ZeroLength,
    #[error("alignment {alignment} is not a nonzero power of two")]
    InvalidAlignment { alignment: usize },
    #[error("address {address} is not aligned to {alignment}")]
    Misaligned { address: Address, alignment: usize },
    #[error("requested length {requested} exceeds limit {limit}")]
    LengthExceeded { requested: usize, limit: usize },
    #[error("range traversal exceeded its {limit}-region limit")]
    RegionLimitExceeded { limit: usize },
    #[error("address {address} is not mapped")]
    Unmapped { address: Address },
    #[error("range at {address} requires {required:?}, but region has {actual:?}")]
    PermissionDenied {
        address: Address,
        required: Access,
        actual: Access,
    },
    #[error("mapping query for {address} did not return a containing, advancing region")]
    NonProgressingRegion { address: Address },
    #[error("read transferred {actual} of {expected} bytes")]
    PartialRead { expected: usize, actual: usize },
    #[error("write transferred {actual} of {expected} bytes")]
    PartialWrite { expected: usize, actual: usize },
    #[error("source range {source_range:?} overlaps destination range {destination:?}")]
    OverlappingRanges {
        source_range: AddressRange,
        destination: AddressRange,
    },
    #[error("string was not terminated within {max_len} bytes")]
    UnterminatedString { max_len: usize },
    #[error("invalid value for {context}")]
    InvalidValue { context: &'static str },
    #[error("{operation:?} is unsupported on this platform")]
    UnsupportedPlatform { operation: PlatformOperation },
    #[error("platform {operation:?} failed with code {code}")]
    Platform {
        operation: PlatformOperation,
        code: i64,
    },
}

pub fn validate_alignment(alignment: usize) -> Result<(), MemoryError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        Err(MemoryError::InvalidAlignment { alignment })
    } else {
        Ok(())
    }
}

/// Supplies point-in-time mapping information for current-process addresses.
pub trait MemoryMap: Send + Sync {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError>;

    fn check_range(
        &self,
        range: AddressRange,
        required: Access,
        max_regions: usize,
    ) -> Result<(), MemoryError> {
        if max_regions == 0 {
            return Err(MemoryError::RegionLimitExceeded { limit: 0 });
        }

        let mut cursor = range.start();
        for _ in 0..max_regions {
            let region = self.region_containing(cursor)?;
            if !region.contains(cursor) || region.range.end() <= cursor {
                return Err(MemoryError::NonProgressingRegion { address: cursor });
            }
            if !region.access.contains(required) {
                return Err(MemoryError::PermissionDenied {
                    address: cursor,
                    required,
                    actual: region.access,
                });
            }
            if region.range.end() >= range.end() {
                return Ok(());
            }
            cursor = region.range.end();
        }

        Err(MemoryError::RegionLimitExceeded { limit: max_regions })
    }
}

/// Direct, in-process byte copying after complete-range preflight.
///
/// # Safety
///
/// Implementations must not construct references from `address`. Callers must
/// keep the full range mapped, permitted, and initialized where read from the
/// beginning of preflight through completion of the copy, and must uphold all
/// raw-pointer synchronization requirements.
pub trait DirectMemoryAccess: MemoryMap {
    unsafe fn read_bytes_into(
        &self,
        address: Address,
        output: &mut [u8],
    ) -> Result<(), MemoryError>;

    unsafe fn write_bytes_from(&self, address: Address, bytes: &[u8]) -> Result<(), MemoryError>;
}

/// A memory backend whose read implementation contains invalid-address faults.
///
/// # Safety
///
/// An implementation must accept arbitrary address/length inputs without
/// causing Rust undefined behavior or a native memory fault. `Ok(())` may be
/// returned only after an exact transfer. Reads are safe; writes remain unsafe
/// because mapping checks cannot establish foreign program invariants.
pub unsafe trait FaultContainedMemoryAccess: MemoryMap {
    fn try_read_bytes_into(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError>;

    unsafe fn try_write_bytes_from(
        &self,
        address: Address,
        bytes: &[u8],
    ) -> Result<(), MemoryError>;
}

#[cfg(any(test, feature = "test-support"))]
pub mod testing;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_address_arithmetic_rejects_overflow() {
        assert_eq!(
            Address::new(usize::MAX).checked_add(1),
            Err(MemoryError::AddressOverflow)
        );
        assert_eq!(
            Address::NULL.checked_sub(1),
            Err(MemoryError::AddressOverflow)
        );
        assert_eq!(
            Address::NULL.checked_add_signed(-1),
            Err(MemoryError::AddressOverflow)
        );
    }

    #[test]
    fn ranges_are_non_empty_and_half_open() {
        assert_eq!(
            AddressRange::new(Address::new(4), 0),
            Err(MemoryError::ZeroLength)
        );
        let range = AddressRange::new(Address::new(4), 4).unwrap();
        assert!(range.contains(Address::new(4)));
        assert!(range.contains(Address::new(7)));
        assert!(!range.contains(Address::new(8)));
    }

    #[test]
    fn alignment_must_be_a_nonzero_power_of_two() {
        assert_eq!(
            Address::new(8).is_aligned_to(0),
            Err(MemoryError::InvalidAlignment { alignment: 0 })
        );
        assert_eq!(
            Address::new(8).is_aligned_to(3),
            Err(MemoryError::InvalidAlignment { alignment: 3 })
        );
        assert!(Address::new(8).is_aligned_to(8).unwrap());
        assert!(!Address::new(9).is_aligned_to(8).unwrap());
    }

    struct Regions(Vec<MemoryRegion>);

    impl MemoryMap for Regions {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            self.0
                .iter()
                .copied()
                .find(|region| region.contains(address))
                .ok_or(MemoryError::Unmapped { address })
        }
    }

    fn region(start: usize, len: usize, access: Access) -> MemoryRegion {
        MemoryRegion {
            range: AddressRange::new(Address::new(start), len).unwrap(),
            access,
            kind: RegionKind::Private,
            module: None,
        }
    }

    #[test]
    fn complete_range_walks_adjacent_regions() {
        let map = Regions(vec![
            region(0x1000, 0x10, Access::READ),
            region(0x1010, 0x10, Access::READ | Access::WRITE),
        ]);
        let requested = AddressRange::new(Address::new(0x1008), 0x18).unwrap();
        assert_eq!(map.check_range(requested, Access::READ, 2), Ok(()));
        assert_eq!(
            map.check_range(requested, Access::WRITE, 2),
            Err(MemoryError::PermissionDenied {
                address: Address::new(0x1008),
                required: Access::WRITE,
                actual: Access::READ,
            })
        );
    }

    #[test]
    fn complete_range_rejects_gaps_and_region_limit() {
        let map = Regions(vec![
            region(0x1000, 0x10, Access::READ),
            region(0x1020, 0x10, Access::READ),
        ]);
        let requested = AddressRange::new(Address::new(0x1008), 0x20).unwrap();
        assert_eq!(
            map.check_range(requested, Access::READ, 2),
            Err(MemoryError::Unmapped {
                address: Address::new(0x1010)
            })
        );

        let adjacent = Regions(vec![
            region(0x1000, 0x10, Access::READ),
            region(0x1010, 0x10, Access::READ),
        ]);
        assert_eq!(
            adjacent.check_range(requested, Access::READ, 1),
            Err(MemoryError::RegionLimitExceeded { limit: 1 })
        );
    }
}
