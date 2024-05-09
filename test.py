import kfac.mischief as mischief

mischief.mischief_init(world_size=16, possible_disconnect_node=None,
                           max_disconnect_iter=20, disconnect_ratio=0.2,
                            max_disconnected_node_num=4,
                           ddp_trigger=True, factor_comm_trigger=True, inverse_comm_trigger=True)

for i in range(1000):
    mischief.update_iter()
    if mischief.ITER >= mischief.LAST_AVG_ITER + 10:
        print(f"update avg at iter {mischief.ITER}")
        mischief.LAST_AVG_ITER = mischief.ITER