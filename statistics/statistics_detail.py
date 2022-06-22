#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: statistics_detail.py
# @Author: Li Chengxin 
# @Time: 2022/6/20 15:16

import os
import sys
from copy import deepcopy
import openpyxl as px


class CellDetail(object):

    def __init__(self, cell_id, start, end):
        self.__template = {
            'start_frame': int,
            'end_frame': int,
            'continue_frame': int,
            'parent_id': int
        }
        self.cell_id = cell_id
        self.cell_type = None
        self.start = start
        self.end = end
        self.G1 = None
        self.S = None
        self.G2 = None
        self.M = None
        self.__set_detail_flag = False
        self.order = []

    def set_details(self, phase, start=None, end=None, parent=None):
        info = deepcopy(self.__template)
        info['phase'] = phase
        info['start_frame'] = start
        info['end_frame'] = end
        info['continue_frame'] = end - start + 1
        info['parent_id'] = parent

        if 'G1' in phase:
            self.G1 = info
        elif 'S' in phase:
            self.S = info
        elif 'G2' in phase:
            self.G2 = info
        elif 'M' in phase:
            self.M = info
        else:
            raise ValueError(phase)
        self.order.append(phase.replace('*', ''))
        self.__set_detail_flag = True
        return info

    def get_details(self):
        if not self.__set_detail_flag:
            return {'id': self.cell_id,
                    'start': self.start,
                    'end': self.end}
        else:
            details = {
                'id': self.cell_id,
                'type': self.cell_type,
                'start': self.start,
                'end': self.end,
                'phase': {'G1': self.G1, 'S': self.S, 'G2': self.G2, 'M': self.M}
            }
        return details

    def sort(self):
        if self.__set_detail_flag:
            return self.order
        else:
            return None

    def __eq__(self, other):
        return self.cell_id == other.cell_id

    def __str__(self):
        cell_type_num = ((self.G1 is not None) + (self.S is not None) + (self.G2 is not None) + (self.M is not None))
        string = f"""
        Object of CellDetail. Exist {cell_type_num} types phase,
        exist frame from {self.start} to {self.end}.
        each cell cycle phase details:
        G1: {self.G1}
        S: {self.S}
        G2: {self.G2}
        M: {self.M}
        """
        return string


class RefinedParser(object):
    def __init__(self, path):
        wb = px.load_workbook(path)
        sheet = wb[wb.sheetnames[0]]
        self.frame_details = sheet['A'][1:]
        self.id_details = sheet['B'][1:]
        self.lineage_details = sheet['C'][1:]
        self.parent_id_details = sheet['D'][1:]
        self.phase_details = sheet['S'][1:]

    def parse_id(self):
        id_record = []
        id_info = []
        current_index = 0
        for i in range(len(self.id_details)):
            if self.id_details[i].value not in id_record:
                id_record.append(self.id_details[i].value)
        for _id in id_record:
            length = 0
            start = current_index
            for j in self.id_details[current_index:]:
                if j.value == _id:
                    length += 1
                else:
                    current_index += length
                    break
            end = start + length
            id_info.append({'id': _id, 'start': start, 'end': end, 'continue': length})
            # print({'id': _id, 'start': start, 'end': end, 'continue': length})
        return id_info

    def parse_phase(self):
        id_info = self.parse_id()
        cells = []
        for i in id_info:
            _id = i['id']
            start_index = i['start']
            end_index = i['end']
            phases = []
            phase_info = []
            for ps in self.phase_details[start_index: end_index]:
                if ps.value not in phases:
                    phases.append(ps.value)
            cell_type = 'normal'
            for pp in phases:
                if '*' in pp:
                    cell_type = pp.replace('*', ' ') + 'arrest'
            current = start_index
            for i in phases:
                length = 0
                start = current
                for j in self.phase_details[start_index: end_index]:
                    if j.value == i:
                        length += 1
                end = start + length
                current = end
                # print([i, start, end, length])
                phase_info.append({'phase': i, 'start': start, 'end': end})
            # print(phase_info)
            cell = CellDetail(cell_id=_id, start=self.frame_details[start_index].value,
                              end=self.frame_details[end_index - 1].value)
            for pi in phase_info:
                # cell.set_details(phase=pi['phase'], start=pi['start'], end=pi['end'])
                cell.set_details(phase=pi['phase'], start=self.frame_details[pi['start']].value,
                                 end=self.frame_details[pi['end'] - 1].value)
                cell.cell_type = cell_type
            cells.append(cell)
        return cells
        # print(cell.get_details())
        # print('------------------')
        # print(start_index)
        # print(end_index)
        # print(self.frame_details[start_index].value)
        # print(self.frame_details[end_index - 1].value)
        # break

    def get_cells_details(self):
        return self.parse_phase()

    def fill_gap(self, cell: CellDetail):
        sort = cell.sort()
        phase_detail = cell.get_details()
        data = {'M1': 0, 'G1': 0, 'S': 0, 'G2': 0, 'M2': 0, 'type': cell.cell_type}
        if sort:
            phase = phase_detail.get('phase')
            if sort[0] == 'M':
                data['M1'] = phase.get('M').get('continue_frame')
            elif ('M' in sort) and len(sort) > 1 and sort[0] != 'M':
                data['M2'] = phase.get('M').get('continue_frame')
            for i in sort:
                if i == 'M':
                    continue
                data[i] = phase.get(i).get('continue_frame')
        return data

    def export_result(self):
        all_data = []
        for i in self.parse_phase():
            data = []
            c = self.fill_gap(i)
            data.append(c.get('M1'))
            data.append(c.get('G1'))
            data.append(c.get('S'))
            data.append(c.get('G2'))
            data.append(c.get('M2'))
            # data.append(c.get('type'))
            all_data.append(data)
        # print(len(self.parse_phase()))
        # print(all_data)
        ret = sorted(all_data, key=lambda List: sum(List), reverse=True)
        # print(ret)
        return all_data, ret


if __name__ == '__main__':
    path = r'G:\20x_dataset\copy_of_xy_16\refined.xlsx'
    # load(path)
    r = RefinedParser(path=path)
    r.export_result()
    # for i in r.get_cells_details():
    #     print(i)
