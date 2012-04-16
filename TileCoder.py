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
    
class MultiTileCoder:

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
                    [ itertools.repeat (self.num_tiles.take(subspace).prod(),
                                        self.num_offsets.take(subspace).prod())
                      for subspace in subspaces ])),
            dtype=np.intp).cumsum()

        self.num_features = ixs_start[-1]
        self.active_features = self.ixs_start.size

    def indices (self, coord):

        num_tiles = self.num_tiles
        num_offsets = self.num_offsets

        offset_start = coord / self.tile_size
        tile_coord = np.floor (offset_coord)

        offset_start -= tile_coord
        offset_start *= offsets
        offset_start = offset_start.astype(np.intp)
        offset_start += 1

        tile_coord = tile_coord.astype(np.intp)
        tile_coord %= tiles

        tile_coord_offset = tile_coord - 1
        tile_coord_offset %= tiles

        ixs = np.zeros (self.active_features, dtype=np.intp)
        ixs_view = ixs.view()

        for subspace in subspaces:

            subspace_offsets = offsets[subspace]
            num_subspace_offsets = subspace_offsets.prod()

            ixs_offsets = ixs_view[:num_subspace_offsets]
            ixs_offsets.shape = subspace_offsets

            ixs_view = ixs_view[num_subspace_offsets:]

            for i in subspace.nonzero():
                
                ixs_offsets *= tiles[i]
                ixs_offsets[:offset_start[i]] += tile_coord[i]
                ixs_offsets[offset_start[i]:] += tile_coord_offset[i]

                ixs_offsets = np.rollaxis (ixs_offsets, 0, ixs_offsets.ndim)

        ixs[1:] += self.ixs_start[:-1]

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
