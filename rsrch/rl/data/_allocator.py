from copy import deepcopy


class Allocator:
    """A helper class for managing (virtual) allocations. Mainly used for
    reserving space in preallocated arrays for RL buffers."""

    def __init__(self, mem_size: int):
        self.slots = {}
        self.mem_size = mem_size
        self.free_space = mem_size
        self.blocks = [[0, mem_size]]

    def malloc(self, id: int, n: int) -> slice | None:
        """Allocate a block of memory with size n and identifier id."""
        if self.free_space < n:
            return

        for idx in range(len(self.blocks)):
            ptr, end = self.blocks[idx]
            if end - ptr >= n:
                self.slots[id] = ptr, ptr + n
                self.blocks[idx][0] = ptr + n
                self.free_space -= n
                return slice(ptr, ptr + n)

    def realloc(self, id: int, new_size: int):
        """Reallocate a block of memory."""
        ptr, end = self.slots[id]
        if end - ptr > new_size:
            self.shrink(id, new_size)
        elif end - ptr < new_size:
            state = deepcopy(self)
            self.free(id)
            blk = self.malloc(id, new_size)
            if blk is None:
                for attr in ["slots", "mem_size", "free_space", "blocks"]:
                    setattr(self, attr, getattr(state, attr))
            return blk

    def shrink(self, id: int, new_size: int):
        """Shrink a block of memory."""
        ptr, end = self.slots[id]
        if end - ptr > new_size:
            self.slots[id] = ptr, ptr + new_size
            self.free_space += (end - ptr) - new_size
            self.blocks.append([ptr + new_size, end])
            self._fix_blk()
            return slice(ptr, ptr + new_size)

    def _fix_blk(self):
        """Normalize blocks structure by sorting it and merging adjacent blocks."""
        self.blocks.sort()
        blocks = []
        for idx in range(len(self.blocks)):
            ptr, end = self.blocks[idx]
            if len(blocks) == 0 or blocks[-1][-1] < ptr:
                blocks.append([ptr, end])
            else:
                blocks[-1][-1] = end
        self.blocks = blocks

    def __getitem__(self, id: int) -> slice | None:
        """Get the extent of memory allocation, given the id."""
        return slice(*self.slots[id]) if id in self.slots else None

    def free(self, id: int):
        """Free allocation with identifier id."""
        ptr, end = self.slots[id]
        self.free_space += end - ptr
        del self.slots[id]
        self.blocks.append([ptr, end])
        self._fix_blk()

    def defrag(self) -> list[tuple[int, slice, slice]]:
        """Squeeze free blocks between allocations, and return a map of
        allocations that were moved, with the source and destination blocks."""

        slots = [(ptr, end, id) for id, (ptr, end) in self.slots.items()]
        slots.sort()

        dst_ptr, map_ = 0, []
        for slot_ptr, slot_end, id in slots:
            src_blk = slice(slot_ptr, slot_end)
            n = src_blk.stop - src_blk.start
            dst_blk = slice(dst_ptr, dst_ptr + n)
            if src_blk != dst_blk:
                map_.append((id, src_blk, dst_blk))
            self.slots[id] = dst_blk.start, dst_blk.stop
            dst_ptr += n

        if dst_ptr < self.mem_size:
            self.blocks = [[dst_ptr, self.mem_size]]
        else:
            self.blocks = []

        return map_

    def clear(self):
        """Free all allocated blocks."""

        self.slots = {}
        self.free_space = self.mem_size
        self.blocks = [[0, self.mem_size]]
