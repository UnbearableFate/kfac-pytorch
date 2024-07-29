from scipy.special import expit


def compute_recv_weight_by_loss(local_loss, recv_loss):
    sigmoid_param = local_loss - recv_loss / local_loss
    return expit(sigmoid_param*0.3)

print(compute_recv_weight_by_loss(1,0.8))