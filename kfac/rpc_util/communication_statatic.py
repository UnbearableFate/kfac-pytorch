
class CommunicationStatics:
    def __init__(self):
        self.send_stat = {}
        self.recv_stat = {}
    def add_send_stat(self, key, times=1):
        if key not in self.send_stat:
            self.send_stat[key] = times
        self.send_stat[key] += times
    def add_recv_stat(self, key, times=1):
        if key not in self.recv_stat:
            self.recv_stat[key] = times
        self.recv_stat[key] += times
    def __repr__(self):
        return "CommunicationStatics(send_stat={}, recv_stat={})".format(self.send_stat, self.recv_stat)
        