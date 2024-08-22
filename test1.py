from scipy.special import expit


def compute_recv_weight_by_loss(local_loss, recv_loss):
    sigmoid_param = local_loss - recv_loss / local_loss
    return expit(sigmoid_param*0.3)


def create_groups(world_size, rank):
    send_rank_group = []
    rank_group_number = -1

    for i in range(0, world_size, 4):
        group = list(range(i, min(i + 4, world_size)))

        if rank in group:
            group.remove(rank)
            rank_group_number = len(send_rank_group)

        send_rank_group.append(group)

    return send_rank_group, rank_group_number


# 示例用法:
world_size = 16
rank = 13
groups, rank_group_number = create_groups(world_size, rank)
print(groups)
print(rank_group_number)