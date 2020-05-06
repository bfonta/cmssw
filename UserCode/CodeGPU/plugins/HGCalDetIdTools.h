namespace detid_tools {
  static constexpr int kHGCalCellOffset = 0;
  static constexpr int kHGCalCellMask = 0xFF;
  static constexpr int kHGCalWaferOffset = 8;
  static constexpr int kHGCalWaferMask = 0x3FF;
  static constexpr int kHGCalWaferTypeOffset = 18;
  static constexpr int kHGCalWaferTypeMask = 0x1;
  static constexpr int kHGCalLayerOffset = 19;
  static constexpr int kHGCalLayerMask = 0x1F;
  static constexpr int kHGCalZsideOffset = 24;
  static constexpr int kHGCalZsideMask = 0x1;
  static constexpr int kHGCalMaskCell = 0xFFFBFF00;

  /// get the absolute value of the cell #'s in x and y
  constexpr int cell(uint32_t id) { return id & kHGCalCellMask; }

  /// get the wafer #
  constexpr int wafer(uint32_t id) { return (id >> kHGCalWaferOffset) & kHGCalWaferMask; }

  /// get the wafer type
  constexpr int waferType(uint32_t id) { return ((id >> kHGCalWaferTypeOffset) & kHGCalWaferTypeMask ? 1 : -1); }

  /// get the layer #
  constexpr int layer(uint32_t id) { return (id >> kHGCalLayerOffset) & kHGCalLayerMask; }

  /// get the z-side of the cell (1/-1)
  constexpr int zside(uint32_t id) { return ((id >> kHGCalZsideOffset) & kHGCalZsideMask ? 1 : -1); }
}
