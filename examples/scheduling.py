"""Persistent output-tile scheduling using the public TIRx API."""

from tvm_tirx.script import tirx as T


@T.meta_class
class PersistentScheduler2D:
    """Group adjacent M tiles for L2 reuse and stride by the resident cluster count."""

    def __init__(self, name, num_m_tiles, num_n_tiles, l2_group_size, num_clusters):
        self.m_tiles = num_m_tiles
        self.n_tiles = num_n_tiles
        self.group_size = l2_group_size
        self.stride = num_clusters
        self.linear_idx = T.bind_(T.local_scalar("int32"))
        self.m_idx = T.bind_(T.local_scalar("int32"))
        self.n_idx = T.bind_(T.local_scalar("int32"))

    @T.inline
    def update(self):
        group = self.linear_idx // (self.group_size * self.n_tiles)
        first_m = group * self.group_size
        # Idle clusters also initialize the cursor, so keep the divisor positive.
        rows = T.max(1, T.min(self.group_size, self.m_tiles - first_m))
        within = self.linear_idx % (self.group_size * self.n_tiles)
        self.m_idx = first_m + within % rows
        self.n_idx = within // rows

    @T.inline
    def init(self, cluster_id):
        self.linear_idx = cluster_id
        self.update()

    def valid(self):
        return self.linear_idx < self.m_tiles * self.n_tiles

    @T.inline
    def next_tile(self):
        self.linear_idx += self.stride
        self.update()
