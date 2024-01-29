import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class MambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int):
        super()
        # First, we instantiate the MLPs
        self.linear_1 = nn.Linear(hidden_dim, mlp_dim)
        self.linear_2 = nn.Linear(hidden_dim, mlp_dim)

        self.output_proj = nn.Linear(mlp_dim, hidden_dim)

        # Next, we instantiate the conv layer
        self.conv = None

        # Finally, we instantiate the ssm
        self.ssm_layer = None
    
    def forward(self, input):
        # input = bs x seqlen x hidden_dim

        assert len(input.shape) == 3, f"The input shape should be three but is {input.shape}"

        
        # Independent per-token
        alternate_path = self.linear_2(input)
        alternate_path = F.sigmoid(alternate_path)

        # Path where token mixing occurs (cross-sequence information)
        mixed_path = self.linear_1(input)
        mixed_path = self.conv(input)
        mixed_path = F.sigmoid(input)
        mixed_path = self.ssm_layer(mixed_path)


        assert mixed_path.shape == alternate_path.shape, \
                f"Mismatch in shapes between mixed path: {mixed_path.shape} and alternate path {alternate_path.shape}"
        
        elemwise_mult_out = mixed_path * alternate_path

        return self.output_proj(elemwise_mult_out)

        
class SSM(nn.Module):
    def __init__(self, d: int, n: int):
        super()
        self.s_b = nn.Linear(d, n, bias=False)
        self.s_c = nn.Linear(d, n, bias=False)

        self.d = d
        self.n = n

        self.s_delta_linear = nn.Linear(d, 1, bias=False)

        self.A = nn.Parameter(torch.randn(d, n))

        self.delta_parameter = nn.Parameter(torch.randn(d))

    def forward(self, input: torch.Tensor):
        # These are completely parallel ops, so they don't need to
        # be handled specially
        B = self.s_b(input)
        C = self.s_c(input)

        # Repeat the last dimension 'd' times
        s_delta_x = self.s_delta_linear(input).repeat_interleave(self.d, dim=-1)
        # TODO - verify that the delta_parameter auto-broadcasts/matches bxl dimensions
        Delta = F.softplus(s_delta_x + self.delta_parameter)

        # Delta is now B x L x D
        # A is D X N (really represents D by N X N)
        unexponentiated_A_bar_diag = torch.zeros(*Delta.shape, self.n)
        # A_bar_diagonalized is B x L x D x N
        for i in range(self.d):
            # 1 x 1 x n
            unsqueezed_A_at_d = self.A[i].unsqueeze(0).unsqueeze(0)

            # I believe pytorch should auto-broadcast the 1 x 1 into B x L (but should verify)
            # TODO - come back to this
            unexponentiated_A_bar_diag[..., i, :] = Delta[..., i].unsqueeze(-1) * unsqueezed_A_at_d
        
        A_bar_diag = torch.exp(unexponentiated_A_bar_diag)

        # # (∆A) −1(exp(∆A) − I)
        # dim = B x L x D x N
        B_bar_lhs_diag = (1 / unexponentiated_A_bar_diag) * (A_bar_diag - 1)

        # # ∆B
        B_bar = torch.zeros(*A_bar_diag.shape)
        for i in range(self.d):
            # B = B x L x N
            # Delta[..., i].unsqueeze(-1) = B x L x 1
            B_bar_rhs = Delta[..., i].unsqueeze(-1) * B

            # Because we are always in diagonalizable matrix land
            # we can just do elemwise products instead of dot products!
            B_bar[..., i, :] = B_bar_lhs_diag[..., i, :] * B_bar_rhs 
        
        # RHS of all the recurrent steps
        b_additions = torch.dot(B_bar, input)

        # 



# Convincing myself of the selective scan
# what is the op s.t.
# (a ± b ) ± c = a ± (b ± c)
# TODO - make it work for the ssm recurrence
class Node:
    value: int
    is_leafnode: bool
    cum_sum: int = None
    parent = None
    left = None
    right = None

    def __init__(self, value: int, is_leafnode: bool):
        self.value = value
        self.is_leafnode = is_leafnode

    def set_children(self, left, right):
        self.left = left
        self.right = right
        self.left.parent = self
        if self.right:
            self.right.parent = self

def upward_pass(elems: List[Node]):
    if len(elems) == 1:
        return

    parent_nodes = []
    # This for loop is parallelizeable
    for i in range(0, len(elems), 2):
        if i == len(elems) - 1:
            new_node = Node(elems[i].value, False)
            new_node.set_children(elems[i], None)
            parent_nodes.append(new_node)
        else:
            new_node = Node(elems[i].value + elems[i+1].value, False)
            new_node.set_children(elems[i], elems[i+1])
            parent_nodes.append(new_node)

    upward_pass(parent_nodes)

def downwards_pass(root_node, value_to_add = 0):
    if root_node.is_leafnode:
        root_node.cum_sum = root_node.value + value_to_add
    else:
        # These two are parallelizable
        downwards_pass(root_node.left, value_to_add)
        # In the odd number case, this does not exist
        if root_node.right:
            downwards_pass(root_node.right, value_to_add + root_node.left.value)
    


def up_down(values: List[int]):
    nodes = [Node(value, True) for value in values]
    upward_pass(nodes)

    new_parent = nodes[0].parent
    parent = nodes[0]

    while new_parent is not None:
        parent = new_parent
        new_parent = parent.parent

    downwards_pass(parent)

    return [node.cum_sum for node in nodes]
    

if __name__ == '__main__':
    result = up_down([1, 2, 4, 9, 0, 4, 8, 1, 20])

    assert result == [1, 3, 7, 16, 16, 20, 28, 29, 49]