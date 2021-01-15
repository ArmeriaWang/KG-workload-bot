import mysql.connector
import random
import datetime
import time



def is_daytime(t):
    """
    判断时间t是否在白天时间
    """
    if 0 <= t.hour <= 7:
        return False
    if t.hour >= 22:
        return False
    return True

# col[0] 表名
# col[1] 列名
# col[2] 数据类型
# 若为INT型，则col[3]和col[4]是合法范围区间
# 若为STR型，则col[3]是常用字符集合，l表示小写字母，n表示数字
def generate_condition(col):
    """
    在列col上生成条件
    """
    if col[2] == 'STR':
        q_char = ' '
        if col[3] == 'l':
            q_char = random.sample(common_str_lower, 1)[0]
        elif col[3] == 'n':
            q_char = random.sample(common_str_number, 1)[0]
        return "{}.{} LIKE '{}%'".format(col[0], col[1], q_char)
    if col[2] == 'INT':
        return "{}.{} <= {}".format(col[0], col[1], random.randint(col[3], col[4]))


def generate_daytime_query(cur_time):
    """
    生成白天的询问
    """
    index_cols = set()
    weekday = cur_time.weekday() - 1
    prob = random.random()
    if prob < 0.9:
        cols = random.sample(weekday_cols[weekday], 2)
        use_one_prob = random.random()
        if use_one_prob < 0.75:
            cols = (cols[0], cols[0])
    else:
        cols = random.sample(tab_col_list, 2)
    if cols[0][0] != cols[1][0]:
        tab1 = cols[0][0]
        tab2 = cols[1][0]
        col1 = cols[0][1]
        col2 = cols[1][1]
        index_cols.add('{}.{}'.format(tab1, col1))
        index_cols.add('{}.{}'.format(tab2, col2))
        query = "SELECT {tab1}._subject_id_, {tab2}._subject_id_ FROM {tab1}, {tab2} WHERE " \
            .format(tab1=tab1, tab2=tab2)
        condition1 = generate_condition(cols[0])
        condition2 = generate_condition(cols[1])
        query += '{} AND {} AND '.format(condition1, condition2)
        if (tab1, tab2) in join_stat:
            prob = random.random()
            if prob < 0:
                stat = random.sample(join_stat[(tab1, tab2)], 1)[0]
            else:
                stat = ("{}._subject_id_".format(tab1), "{}._subject_id_".format(tab2))
            query += '{}={} '.format(stat[0], stat[1])
            index_cols.add(stat[0])
            index_cols.add(stat[1])
        query += 'limit 100000'
    else:
        tab = cols[0][0]
        col1 = cols[0][1]
        col2 = cols[1][1]
        index_cols.add('{}.{}'.format(tab, col1))
        index_cols.add('{}.{}'.format(tab, col2))
        query = "SELECT * FROM {} WHERE ".format(tab)
        condition1 = generate_condition(cols[0])
        condition2 = generate_condition(cols[1])
        query += '{} AND {}'.format(condition1, condition2)
    return query, index_cols


def generate_nighttime_query(is_insert):
    """
    生成夜间的询问，
    若is_insert为True，生成插入
    若is_insert为False，生成删除
    """
    ins_tab = random.sample(tab_cols.keys(), 1)[0]
    index_cols = ["{}._subject_id_".format(ins_tab)]
    if is_insert:
        return "CREATE TEMPORARY TABLE tmp_table_1 SELECT * FROM {tab} WHERE _subject_id_ = 1; " \
               "UPDATE tmp_table_1 SET _subject_id_ = 0; " \
               "INSERT INTO {tab} SELECT * FROM tmp_table_1; " \
               "DROP TEMPORARY TABLE IF EXISTS tmp_table_1;".format(tab=ins_tab), index_cols
    else:
        return "DELETE FROM {} ORDER BY _subject_id_ DESC limit 1".format(ins_tab), index_cols


def create_queries(start_time, end_time, time_interval=60 * 5):
    """
    生成从start_time到end_time的历史负载，每两个询问的间隔为300秒
    """
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345678"
    )

    cursor = conn.cursor(buffered=True)
    cursor.execute("USE {}".format("main"))

    # start_time = datetime.datetime(2021, 1, 1, 0, 0)
    # end_time = datetime.datetime(2021, 7, 1, 0, 0)
    cur_time = start_time
    time_interval = datetime.timedelta(seconds=time_interval)
    q_file = './iofiles/query_file_{}.txt'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    col_cnt = {}
    is_insert = True
    with open(q_file, 'w') as f:
        while True:
            if is_daytime(cur_time):
                query, index_cols = generate_daytime_query(cur_time)
            else:
                query, index_cols = generate_nighttime_query(is_insert)
                is_insert = not is_insert
            index_cols_str = ""
            for col in index_cols:
                if col.endswith('_subject_id_'):
                    continue
                col_cnt[(cur_time.weekday(), col)] = col_cnt.get((cur_time.weekday(), col), 0) + 1
                index_cols_str += col + ' '
            f.write("{}\n".format(cur_time))
            f.write("{}\n".format(query))
            f.write("{}\n".format(index_cols_str))
            # print(cur_time)
            # print(query)
            # print(index_cols_str)
            cur_time += time_interval
            if cur_time >= end_time:
                break
        f.flush()
    f.close()

    return q_file


if __name__ == '__main__':
    # 生成模拟工作负载，保存在txt文件中
    common_str_lower = {'k', 'li', 'st', 'la', 'ai', 'ae', 'ea', 'il', 'ia', 'ga', 'ma', 'en', 'an', 'al', 'et', 'ee',
                        'se',
                        'es', 'ca', 'me', 'ou', 'na', 'le', 'co', 'll', 'cu', 'th', 'ra', 'ar', 'ay'}
    common_str_number = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    tab_cols = {'ib_album': [('_subject_id_', 'INT', 1, 53927),
                             ('_subject_', 'STR', 'l'),
                             ('P_genre', 'STR', 'l'),
                             ('P_type', 'STR', 'l'),
                             ('P_artist', 'STR', 'l'),
                             ('P_producer', 'STR', 'l'),
                             ('P_reviews', 'STR', 'l'),
                             ('P_name', 'STR', 'l'),
                             ('P_cover', 'STR', 'l'),
                             ('P_length', 'STR', 'n'),
                             # ('P_language', 'STR', 'l')
                             ],
                'ib_film': [('_subject_id_', 'INT', 1, 23201),
                            ('_subject_', 'STR', 'l'),
                            ('P_music', 'STR', 'l'),
                            ('P_language', 'STR', 'l'),
                            ('P_writer', 'STR', 'l'),
                            ('P_producer', 'STR', 'l'),
                            ('P_distributor', 'STR', 'l'),
                            ('P_name', 'STR', 'l'),
                            ('P_director', 'STR', 'l'),
                            ('P_imdbId', 'INT', 1, 4455915)],
                "ib_musical_artist": [('_subject_id_', 'INT', 1, 15148),
                                      ('_subject_', 'STR', 'l'),
                                      ('P_genre', 'STR', 'l'),
                                      ('P_yearsActive', 'STR', 'n'),
                                      ('P_background', 'STR', 'l'),
                                      ('P_currentMembers', 'STR', 'l'),
                                      ('P_name', 'STR', 'l'),
                                      ('P_label', 'STR', 'l'),
                                      ('P_url', 'STR', 'l')],
                "ib_language": [('_subject_id_', 'INT', 1, 975),
                                ('_subject_', 'STR', 'l'),
                                ('P_familycolor', 'STR', 'l'),
                                ('P_fam', 'STR', 'l'),
                                ('P_name', 'STR', 'l'),
                                ('P_iso', 'STR', 'l')],
                "ib_person": [('_subject_id_', 'INT', 1, 1081),
                              ('_subject_', 'STR', 'l'),
                              ('P_image', 'STR', 'l'),
                              ('P_name', 'STR', 'l'),
                              ('P_birthPlace', 'STR', 'l'),
                              ('P_occupation', 'STR', 'l')],
                }

    column_clusters = [
        {'ib_album': ['P_artist', 'P_producer'],
         'ib_film': ['P_music', 'P_producer', 'P_director', 'P_writer'],
         'ib_musical_artist': ['_subject_', 'P_name'],
         'ib_person': ['_subject_', 'P_name']},

        {'ib_album': ['_subject_', 'P_name'],
         'ib_film': ['_subject_', 'P_name']},

        {'ib_film': ['P_language'],
         'ib_language': ['_subject_', 'P_name']},

        {'ib_album': ['P_genre'],
         'ib_musical_artist': ['P_genre']},

        {'ib_album': ['_subject_id_'],
         'ib_film': ['_subject_id_'],
         'ib_musical_artist': ['_subject_id_'],
         'ib_language': ['_subject_id_'],
         'ib_person': ['_subject_id_']}
    ]

    tab_col_list = []
    for key, value in tab_cols.items():
        for col in value:
            tab_col_list.append((key,) + col)
    # print(len(tab_col_list))

    random.shuffle(tab_col_list)
    weekday_cols = []
    for i in range(7):
        weekday_cols.append([])
    for i in range(len(tab_col_list)):
        weekday_cols[i % 7].append(tab_col_list[i])

    join_stat = {}
    for dct in column_clusters:
        for t1 in dct.keys():
            for t2 in dct.keys():
                if t1 == t2:
                    continue
                if not (t1, t2) in join_stat:
                    join_stat[(t1, t2)] = []
                for c1 in dct[t1]:
                    for c2 in dct[t2]:
                        join_stat[(t1, t2)].append(('{}.{}'.format(t1, c1), '{}.{}'.format(t2, c2)))
    # print(join_stat)
