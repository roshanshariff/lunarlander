import numpy as np
import itertools


class TileCoder:

    def __init__ (self, tile_size, num_tiles, num_offsets, subspace_dims, weight_exponent=0.0):

        self.tile_size = np.array (tile_size, dtype=np.float64)
        self.num_tiles = np.array (num_tiles, dtype=np.intp)
        self.num_offsets = np.array (num_offsets, dtype=np.intp)
        self.subspace_dims = subspace_dims

        def subspace_generator ():
            space_dim = len(tile_size)
            for subspace_dim in subspace_dims:
                for ixs in itertools.combinations (xrange(0,space_dim), subspace_dim):
                    ixs = np.array (ixs, dtype=np.intp)
                    tilings = self.num_offsets.take(ixs)
                    num_tilings = tilings.prod()
                    yield (ixs, tilings, num_tilings)

        self.subspaces = list(subspace_generator())
        subspace_tilings = [ s[2] for s in self.subspaces ]
        self.active_features = sum (subspace_tilings)

        def repeat_for_tilings (iterable):
            return itertools.chain.from_iterable (itertools.imap (itertools.repeat, iterable, subspace_tilings))

        tiles_in_tiling = list(repeat_for_tilings ([ self.num_tiles.take(s[0]).prod() for s in self.subspaces ]))
        self.num_features = sum (tiles_in_tiling)
        self.features_start = np.cumsum ([0]+tiles_in_tiling[:-1], dtype=np.intp)

        feature_weights = np.array(subspace_tilings) ** (-weight_exponent)
        self.feature_weights = np.array (list (repeat_for_tilings (feature_weights)), dtype=np.float64)
        self.feature_weights /= np.linalg.norm (self.feature_weights)

    def indices (self, coord):

        num_tiles = self.num_tiles

        offset_start = coord / self.tile_size
        tile_coord = np.floor (offset_start)

        offset_start -= tile_coord
        offset_start *= self.num_offsets
        offset_start = offset_start.astype(np.intp)
        offset_start += 1

        tile_coord = tile_coord.astype(np.intp)
        tile_coord %= num_tiles

        tile_coord_offset = tile_coord - 1
        tile_coord_offset %= num_tiles

        features = np.zeros (self.active_features, dtype=np.intp)
        features_done = 0

        for (ixs, tilings, num_tilings) in self.subspaces:

            subspace_features = features[features_done:features_done+num_tilings]
            subspace_features.shape = tilings
            features_done += num_tilings

            for i in ixs:
                
                subspace_features *= num_tiles[i]
                subspace_features[:offset_start[i]] += tile_coord[i]
                subspace_features[offset_start[i]:] += tile_coord_offset[i]

                subspace_features = np.rollaxis (subspace_features, 0, subspace_features.ndim)

        features += self.features_start
        return features


class OldTileCoder:
    
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

                tiling = self.Tiling (cell_size, subspace_cells, cell_size*offset/subspace_samples)
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

    
class TravisTileCoder: # 18

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
            multi_tiling = self.MultiTiling(cell_size, subspace_cells, offsets)

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
            ixs[i: i + multi_tiling.num_tilings] = multi_tiling.indices(coord)
            i += multi_tiling.num_tilings
        return ixs + self.ix_offsets


class HashingTileCoder:

    def __init__ (self, tile_coder, num_features):
        self.tile_coder = tile_coder
        self.active_features = tile_coder.active_features
        self.feature_weights = tile_coder.feature_weights
        self.num_features = min(num_features, tile_coder.num_features)
        self.hash_const = 2654435761 if num_features < tile_coder.num_features else 1

    def indices (self, coord):
        ixs = self.tile_coder.indices(coord)
        ixs *= self.hash_const
        ixs %= self.num_features
        return ixs

def test_tile_coder (tc, trials):
    oldtc = OldTileCoder (tc.tile_size, tc.num_tiles, tc.num_offsets, tc.subspace_dims)
    travistc = TravisTileCoder (tc.tile_size, tc.num_tiles, tc.num_offsets, tc.subspace_dims)
    max_state = tc.tile_size * tc.num_tiles
    for i in xrange(trials):
        state = max_state * np.random.random(max_state.shape)
        features0 = tc.indices(state)
        features1 = oldtc.indices(state)
        features2 = travistc.indices(state)
        if not np.all(features0 == features1):
            print 'Mismatch on Test 1'
            break
        if not np.all(features0 == features2):
            print 'Mismatch on Test 2'
            break
    print 'Completed'
