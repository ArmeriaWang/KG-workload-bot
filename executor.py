import mysql.connector
import random
import pandas as pd
import numpy as np
import datetime
import time
import argparse

from utils import *

from predictor import predict_next_week_arrivals

config = {'db_name': "main"}
dbconfig = {'index_sql': """SELECT DISTINCT INDEX_NAME, TABLE_NAME FROM INFORMATION_SCHEMA.STATISTICS
                        where table_schema='{}'""",
            'create_index': "CREATE INDEX {} ON {}({})",
            'drop_index': "DROP INDEX {} ON {}"}


def drop_all_indexes():
    """
    丢弃全部索引
    """
    query = dbconfig['index_sql'].format(config['db_name'])
    cursor.execute(query)
    conn.commit()
    indexes = cursor.fetchall()
    for index in indexes:
        index_name = index[0]
        table_name = index[1]
        if index_name.upper() != 'PRIMARY':
            query = dbconfig['drop_index'].format(index_name, table_name)
        else:
            continue

        try:
            cursor.execute(query)
            print("Successfully drop index: ", query)
        except mysql.connector.Error as err:
            print(query)
            print(err)
            continue


def process_day(day_id, start_qid, sim_queries):
    """
    处理每天的工作负载
    """
    day_process_time = 0
    for qid in range(start_qid, len(sim_queries)):
        q_time, query = sim_queries[qid]
        start_qid = qid
        if q_time.date() != start_time.date() + datetime.timedelta(days=day_id):
            break
        qt_start = time.time()
        if query.count(';') > 1:
            cursor.execute(query, multi=True)
        else:
            # print(query)
            cursor.execute(query)
        conn.commit()
        query_process_time = time.time() - qt_start
        day_process_time += query_process_time
        if qid % 10 == 0:
            print("queries {}/{} finished, current day spent {}".format(qid, len(sim_queries), day_process_time))
    print("day: {}\nprocess time={}\n".format(day_id, day_process_time))
    return day_process_time, start_qid


def simulate_prediction():
    """
    运行KGWB的预测模块
    """
    drop_all_indexes()

    replace_index_num = 4
    index_col = []
    index_id = 0

    start_qid = 0
    total_start_time = time.time()
    process_time_seq = []
    days_index_size = []
    for day_id in range(simulate_day_num):
        if len(index_col) + replace_index_num > max_index_num:
            drop_num = len(index_col) + replace_index_num - max_index_num
            drop_index = random.sample(index_col, drop_num)
            for index in drop_index:
                tab_col, name = index.split('#')
                tab, col = tab_col.split('.')
                cursor.execute(dbconfig['drop_index'].format(name, tab))
                index_col.remove(index)
                conn.commit()
        create_index_col = pred_result.nlargest(max_index_num, str(day_id)).index.values.tolist()
        for index_tab_col in create_index_col:
            tab, col = index_tab_col.split('.')
            if col == '_subject_id_':
                continue
            for existed_index in index_col:
                if existed_index.startswith(index_tab_col):
                    continue
            index_name = "index_{}".format(index_id)
            cursor.execute(dbconfig['create_index'].format(index_name, tab, col))
            conn.commit()
            index_id += 1
            index_col.append("{}#{}".format(index_tab_col, index_name))
            if len(index_col) == max_index_num:
                break
        print(index_col)

        day_process_time, start_qid = process_day(day_id, start_qid, sim_queries)
        process_time_seq.append(day_process_time)
        days_index_size.append(get_data_index_size(cursor))

    print(process_time_seq)
    total_time_consuming = time.time() - total_start_time
    print(total_time_consuming)
    days_index_size = np.array(days_index_size)
    print(days_index_size.mean(axis=1))
    print(days_index_size)
    return process_time_seq, total_time_consuming


def simulate_static():
    """
    运行用于对比的静态算法
    """

    drop_all_indexes()
    print(get_data_index_size(cursor))
    col_arrival_cnt = {}
    for i in range(0, len(lines), 3):
        cols = lines[i + 2].split(' ')
        for col in cols:
            if len(col.strip()) > 0:
                col_arrival_cnt[col] = col_arrival_cnt.get(col, 0) + 1

    col_arrival_cnt = pd.Series(col_arrival_cnt).sort_values(ascending=False)
    print(col_arrival_cnt)
    index_cols = col_arrival_cnt.nlargest(max_index_num).index.values.tolist()
    print(index_cols)

    for i in range(len(index_cols)):
        index_name = 'index_{}'.format(i)
        tab, col = index_cols[i].split('.')
        cursor.execute(dbconfig['create_index'].format(index_name, tab, col))
        conn.commit()

    print(get_data_index_size(cursor))

    start_qid = 0
    process_time_seq = []
    total_start_time = time.time()
    for day_id in range(simulate_day_num):
        day_process_time, start_qid = process_day(day_id, start_qid, sim_queries)
        process_time_seq.append(day_process_time)
    print(process_time_seq)
    total_time_consuming = time.time() - total_start_time
    print(total_time_consuming)
    return process_time_seq, total_time_consuming


if __name__ == '__main__':

    argparer = argparse.ArgumentParser(description='Run KGWB executor')
    argparer.add_argument('--run_all', '-a', help='Run all the modules from start', default=False)
    args = vars(argparer.parse_args())
    need_new_query = args['run_all']

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678"
    )
    cursor = conn.cursor(buffered=True)
    cursor.execute("use {}".format(config['db_name']))

    start_time = datetime.datetime(2021, 1, 29, 0, 0)
    end_time = datetime.datetime(2021, 2, 5, 0, 0)
    simulate_day_num = 7
    max_index_num = 5

    newest_time_file = 'newest_files.txt'

    if need_new_query:
        query_file, pred_result_file = predict_next_week_arrivals()
        with open(newest_time_file, 'w') as f:
            f.write("{}\n{}".format(query_file, pred_result_file))

    with open(newest_time_file, 'r') as f:
        query_file, pred_result_file = f.readlines()
        query_file, pred_result_file = query_file.strip(), pred_result_file.strip()

    pred_result = pd.read_csv(pred_result_file, index_col=0)

    with open(query_file, 'r') as f:
        lines = f.readlines()

    sim_queries = []
    for i in range(0, len(lines), 3):
        q_time = get_datetime(lines[i])
        if q_time < start_time:
            continue
        if q_time >= end_time:
            break
        sim_queries.append((q_time, lines[i + 1]))

    pred_time_list, pred_time_tot = simulate_prediction()
    stat_time_list, stat_time_tot = simulate_static()
    print(pred_time_tot, stat_time_tot)
    draw_time_consuming_pic(pred_time_list, stat_time_list)
