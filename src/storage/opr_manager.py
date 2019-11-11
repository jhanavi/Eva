from collections import OrderedDict


class OperationsManager:
    def __init__(self):
        self.opr_id_frame_id_map = OrderedDict()
        self.opr_id_opr_map = {}
        self.next_id = 0

    def add_opr(self, opr, frame_ids):
        opr.id = self.next_id
        self.opr_id_opr_map[opr.id] = opr
        self.opr_id_frame_id_map[opr.id] = frame_ids
        self.next_id += 1

    def del_opr(self, opr_id):
        pass

    def get_opr_list(self, frame_id):
        opr_list = []
        for opr_id, f_ids in self.opr_id_frame_id_map.items():
            if frame_id in f_ids:
                opr_list.append(self.opr_id_opr_map[opr_id])
        return opr_list

