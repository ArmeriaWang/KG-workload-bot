import datetime
from matplotlib import pyplot as plt
from graphviz import Digraph
import torch


def get_datetime(dt):
    q_date, q_clock = dt.split(' ')
    q_year, q_month, q_day = q_date.split('-')
    q_hour, q_minute, q_second = q_clock.split(':')
    # print(q_year, q_month, q_day, q_hour, q_minute, q_second)
    q_time = datetime.datetime(int(q_year), int(q_month), int(q_day), int(q_hour), int(q_minute), int(q_second))
    return q_time


def draw_time_consuming_pic(auto_pred, static):
    days = range(len(auto_pred))
    plt.ylabel('Time Consuming (sec)')
    plt.xlabel('Day')
    plt.bar(days, static, width=0.3, label='static')
    plt.bar([i + 0.3 for i in days], auto_pred, width=0.3, label='KGWBot')
    plt.xticks([i + 0.15 for i in days], days)
    plt.legend(loc='best')
    plt.show()


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
      var: output Variable
      params: dict of (name, Variable) to add names to node that
        require grad (TODO: make optional)
    """
    if params is not None:
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot

def get_data_index_size(cursor):
    cmd_show_index_size = "SELECT CONCAT(table_schema,'.',table_name) AS 'Table Name', " \
                          "CONCAT(ROUND(data_length/(1024*1024),4),'') AS 'Data Size', " \
                          "CONCAT(ROUND(index_length/(1024*1024),4),'') AS 'Index Size', CONCAT(ROUND((" \
                          "data_length+index_length)/(1024*1024),4),'') AS'Total'FROM information_schema.TABLES " \
                          "WHERE table_schema='main'; "
    cursor.execute(cmd_show_index_size)
    col_data_index_len = cursor.fetchall()
    tab_set = {'ib_album',
               'ib_film',
               'ib_musical_artist',
               'ib_language',
               'ib_person'}
    data_index_size_list = []
    total_index_size = 0
    total_data_size = 0
    for item in col_data_index_len:
        item = (item[0], float(item[1]), float(item[2]), float(item[3]))
        db, tab = item[0].split('.')
        for t in tab_set:
            if tab.endswith(t):
                total_data_size += item[1]
                total_index_size += item[2]
                data_index_size_list.append([item[1], item[2], item[3], item[2] / item[3]])
    data_index_size_list.append([total_data_size, total_index_size,
                                 total_data_size + total_index_size,
                                 total_index_size / (total_index_size + total_data_size)])
    return data_index_size_list

