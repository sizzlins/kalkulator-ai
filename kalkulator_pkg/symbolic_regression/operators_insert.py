
def insert_mutation(
    tree: ExpressionTree, operators: list[str] | None = None
) -> ExpressionTree:
    """Insert mutation: Insert a random unary operator above a random node.
    
    This allows evolving nested structures like floor(expr) from expr.
    
    Args:
        tree: Tree to mutate
        operators: Allowed operators
        
    Returns:
        Mutated tree (new copy)
    """
    if operators is None:
        operators = ["sin", "cos", "exp", "square", "floor", "trunc"]
        
    unary_ops = [op for op in operators if op in UNARY_OPERATORS]
    if not unary_ops:
        return tree.copy()
        
    new_tree = tree.copy()
    
    # Pick a random node to wrap
    target = new_tree.get_random_node()
    
    # Create new parent node
    new_op = random.choice(unary_ops)
    new_node = ExpressionNode(
        node_type=NodeType.UNARY_OP, 
        value=new_op, 
        children=[target.copy_subtree()] # Copy target content to be child
    )
    
    # Replace target with new_node
    if target.parent:
        if target.parent.children[0] is target:
            target.parent.children[0] = new_node
        else:
            target.parent.children[1] = new_node
        new_node.parent = target.parent
    else:
        # Target was root
        new_tree.root = new_node
        new_node.parent = None
        
    return new_tree
