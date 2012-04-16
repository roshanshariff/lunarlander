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

class TileCoder:
    
    def __init__ (self, cell_size, num_cells, num_samples, subspace_dims):

        cell_size = np.array(cell_size, dtype=np.float64)
        num_cells = np.array(num_cells, dtype=np.int32)
        num_samples = np.array(num_samples, dtype=np.int32)

        (self.cell_size, self.num_cells, self.num_offsets, self.subspace_dims) = (cell_size, num_cells, num_samples, subspace_dims)

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
        self.strides = np.empty((1,self.num_cells.shape[1]))
        stride = 1
        for i in reversed(xrange(self.strides.shape[1])):
            self.strides[0,i] = stride
            stride *= self.num_cells[0,i]

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

class TravisTileCoder: # 18

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

        # Precompute feature offsets
        self.ix_offsets = np.empty(self.active_features, dtype=int)
        offset = 0
        i = 0
        for mt in self.multi_tilings:
            for n in xrange(0,mt.num_tilings):
                self.ix_offsets[i] = offset
                offset += mt.tiles_per_tiling
                i += 1

    def indices(self, coord):
        ixs = np.empty(self.active_features, dtype=np.int)
        i = 0
        for multi_tiling in self.multi_tilings:
            ixs[i: i + multi_tiling.num_tilings] = multi_tiling.indices(coord) + self.ix_offsets[i: i + multi_tiling.num_tilings]
            i += multi_tiling.num_tilings
        return ixs

class RoshanTileCoder:

    def __init__ (self, tile_size, num_tiles, num_offsets, subspace_dims):

        self.tile_size = np.array (tile_size, dtype=np.float64)
        self.num_tiles = np.array (num_tiles, dtype=np.intp)
        self.num_offsets = np.array (num_offsets, dtype=np.intp)

        space_dim = len(tile_size)

        def select_subspace (ixs):
            subspace = np.zeros (space_dim, dtype=np.bool)
            subspace.put (ixs, True)
            return subspace

        self.subspaces = np.array(
            map (select_subspace,
                 itertools.chain.from_iterable (
                    [ itertools.combinations (range(0, space_dim), dim)
                      for dim in subspace_dims ])),
            dtype=np.bool)

        self.ixs_start = np.array(
            list (itertools.chain.from_iterable (
                    [ itertools.repeat (self.num_tiles[subspace].prod(),
                                        self.num_offsets[subspace].prod())
                      for subspace in self.subspaces ])),
            dtype=np.intp).cumsum()

        self.num_features = self.ixs_start[-1]
        self.active_features = self.ixs_start.size

    def indices (self, coord):

        num_tiles = self.num_tiles
        num_offsets = self.num_offsets

        offset_start = coord / self.tile_size
        tile_coord = np.floor (offset_start)

        offset_start -= tile_coord
        offset_start *= num_offsets
        offset_start = offset_start.astype(np.intp)
        offset_start += 1

        tile_coord = tile_coord.astype(np.intp)
        tile_coord %= num_tiles

        tile_coord_offset = tile_coord - 1
        tile_coord_offset %= num_tiles

        ixs = np.zeros (self.active_features, dtype=np.intp)
        ixs_view = ixs.view()

        for subspace in self.subspaces:

            subspace_offsets = num_offsets[subspace]
            num_subspace_offsets = subspace_offsets.prod()

            ixs_offsets = ixs_view[:num_subspace_offsets]
            ixs_offsets.shape = subspace_offsets

            ixs_view = ixs_view[num_subspace_offsets:]

            for i in subspace.nonzero()[0]:
                
                ixs_offsets *= num_tiles[i]
                ixs_offsets[:offset_start[i]] += tile_coord[i]
                ixs_offsets[offset_start[i]:] += tile_coord_offset[i]

                ixs_offsets = np.rollaxis (ixs_offsets, 0, ixs_offsets.ndim)

        ixs[1:] += self.ixs_start[:-1]
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

def test_tile_coder (oldtc, trials):
    newtc1 = RoshanTileCoder (oldtc.cell_size, oldtc.num_cells, oldtc.num_offsets, oldtc.subspace_dims)
    newtc2 = TravisTileCoder (oldtc.cell_size, oldtc.num_cells, oldtc.num_offsets, oldtc.subspace_dims)
    max_state = oldtc.cell_size * oldtc.num_cells
    for i in xrange(trials):
        state = np.random.random(max_state.shape) * max_state
        old_features = oldtc.indices(state)
        new_features1 = newtc1.indices(state)
        new_features2 = newtc2.indices(state)
        if not np.all(old_features == new_features1):
            print 'Mismatch on TC 1'
            break
        if not np.all(old_features == new_features2):
            print 'Mismatch on TC 2'
            break
    print 'Completed'
