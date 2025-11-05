class Node:
    def __init__(self):
        self.children = None
        self.data = None

    def is_leaf(self) -> bool:
        """
        Checks if the current node is a leaf node.
        A node is a leaf if it has no children.
        """
        return self.children is None

class Octree:
    """
    A basic recursive Octree.

    Each node is either a leaf node containing data or an internal node
    with eight children. The children are ordered to correspond with the
    Z-order curve.
    """

    def __init__(self, depth: int):
        """
        Initializes the Octree recursively to a specified depth.

        Args:
            depth (int): The depth of the tree. If depth is 0, this node is a leaf.
                         If depth > 0, this node is an internal node with 8 children,
                         each initialized with a depth of `depth - 1`.
        """
        self.depth = depth
        self.node = Node()
        

        if depth > 0:
            # This is an internal node. It has 8 children.
            # The children are indexed 0-7. This indexing scheme allows for mapping
            # spatial coordinates (x, y, z) to a child index using Z-order curve
            # logic, e.g., child_index = 4*z_bit + 2*y_bit + 1*x_bit.
            self.children = [Octree(depth - 1) for _ in range(8)]
        else:
            # This is a leaf node (depth == 0). It contains the data.
            # As per instructions, it holds a list of floats.
            self.data = []

    def is_leaf(self) -> bool:
        """
        Checks if the current node is a leaf node.
        A node is a leaf if it has no children.
        """
        return self.children is None

    def __repr__(self) -> str:
        """
        Provides a string representation of the Octree node.
        """
        if self.is_leaf():
            # Assuming data contains floats, show the count.
            return f"Octree(Leaf, data_count={len(self.data)})"
        else:
            return f"Octree(Node, depth={self.depth})"
        
    def Insert(self, point: tuple[float, float, float], data: float):
        """
        Inserts a point into the Octree.
        """
        if self.is_leaf():
            self.data.append(data)
        else:



def EmptyOctree(depth: int) -> Octree:
    """
    Creates an empty Octree of a specified depth.
    """
    return Octree(depth)


def CreateOctree(depth: int) -> Octree:
    """
    Creates an Octree of a specified depth.
    """
    octree = EmptyOctree(depth)

    return octree




