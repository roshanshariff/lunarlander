import numpy as np
import itertools

class Tiling:

    def __init__ (self, cell_size, num_cells, offset):
        self.cell_size = np.array(cell_size, dtype=np.float64)
        self.num_cells = np.array(num_cells, dtype=np.int32)
        self.offset = np.array(offset, dtype=np.float64)
        self.num_tiles = np.prod(num_cells)

    def index (self, coord):
        coord = np.floor((coord - self.offset) / self.cell_size).astype(np.int32)
        coord %= self.num_cells
        return np.ravel_multi_index(coord, self.num_cells)

class OldTileCoder: # 54
    
    def __init__ (self, cell_size, num_cells, num_samples, subspace_dims):

        cell_size = np.array(cell_size, dtype=np.float64)
        num_cells = np.array(num_cells, dtype=np.int32)
        num_samples = np.array(num_samples, dtype=np.int32)

        space_dim = len(cell_size)

        subspaces = itertools.imap (
            lambda s: np.array(s, dtype=np.int32),
            itertools.chain.from_iterable(
                [ itertools.combinations (range(0, space_dim), dim)
                  for dim in subspace_dims ]))

        self.tilings = []
        self.num_features = 0
        self.active_features = 0

        for subspace in subspaces:

            subspace_cells = np.ones_like (num_cells)
            subspace_cells[subspace] = num_cells[subspace]

            subspace_samples = np.ones_like (num_samples)
            subspace_samples[subspace] = num_samples[subspace]

            for offset in itertools.product(*[xrange(0, n) for n in subspace_samples]):

                tiling = Tiling (cell_size, subspace_cells, cell_size*offset/subspace_samples)
                self.tilings.append (tiling)
                self.num_features += tiling.num_tiles
                self.active_features += 1

    def indices (self, coord):
        ixs = np.empty(self.active_features, dtype=int)
        ix_offset = 0
        for (i,tiling) in enumerate(self.tilings):
            ixs[i] = ix_offset + tiling.index(coord)
            ix_offset += tiling.num_tiles
        return ixs

class MultiTiling:
    def __init__(self, cell_size, num_cells, offsets):
        self.cell_size = np.array(cell_size, dtype=np.float64).reshape((1,-1))
        self.num_cells = np.array(num_cells, dtype=np.int32).reshape((1,-1))
        self.offsets = np.array(offsets, dtype=np.float64)

        # Precompute the multiplications required to ravel the index
        stride = 1
        strides = []
        for dim_cells in num_cells:
            strides.append(stride)
            stride *= dim_cells
        self.strides = np.array(strides).reshape((1,-1))

        self.tiles_per_tiling = np.prod(num_cells)
        self.num_tilings = self.offsets.shape[0]
        self.total_tiles = self.tiles_per_tiling * self.num_tilings

    def indices(self, coord):
        coord = coord.reshape((1,-1)) # make into a row vector for broadcasting
        tile_indices = np.floor((coord - self.offsets) / self.cell_size).astype(np.int32)
        tile_indices %= self.num_cells
        tile_indices *= self.strides
        indices = np.sum(tile_indices, axis=1)
        return indices.reshape((-1))

        
class TileCoder: # 18
    def __init__(self, cell_size, num_cells, num_samples, subspace_dims):
        cell_size = np.array(cell_size, dtype=np.float64)
        num_cells = np.array(num_cells, dtype=np.int32)
        num_samples = np.array(num_samples, dtype=np.int32)

        space_dim = len(cell_size)

        subspaces = itertools.imap (
            lambda s: np.array(s, dtype=np.int32),
            itertools.chain.from_iterable(
                [ itertools.combinations (range(0, space_dim), dim)
                  for dim in subspace_dims ]))

        self.multi_tilings = []
        self.num_features = 0
        self.active_features = 0

        for subspace in subspaces:
            subspace_cells = np.ones_like (num_cells)
            subspace_cells[subspace] = num_cells[subspace]

            subspace_samples = np.ones_like (num_samples)
            subspace_samples[subspace] = num_samples[subspace]
            
            offsets = [cell_size * offset / subspace_samples for offset in 
                       itertools.product(*[xrange(n) for n in subspace_samples])]
            multi_tiling = MultiTiling(cell_size, subspace_cells, offsets)

            self.multi_tilings.append(multi_tiling)
            self.num_features += multi_tiling.total_tiles
            self.active_features += multi_tiling.num_tilings

    def indices(self, coord):
        ixs = np.empty(self.active_features, dtype=int)
        ix_offset = 0
        i = 0
        for multi_tiling in self.multi_tilings:
            for ix in multi_tiling.indices(coord):
                ixs[i] = ix_offset + ix
                ix_offset += multi_tiling.tiles_per_tiling
                i += 1
        return ixs

class HashingTileCoder:

    def __init__ (self, tile_coder, num_features):
        self.tile_coder = tile_coder
        self.active_features = tile_coder.active_features
        self.num_features = min(num_features, tile_coder.num_features)
        self.hash_const = 2654435761 if num_features < tile_coder.num_features else 1

    def indices (self, coord):
        ixs = self.tile_coder.indices(coord)
        ixs *= self.hash_const
        ixs %= self.num_features
        return ixs
