# %%

import mysql.connector
import rdflib
from rdflib.namespace import XSD
from rdflib import Graph, Literal


def get_type_xsd2mysql_dict():
    type_xsd2mysql_dict = {
        XSD.string: "VARCHAR(511)", XSD.integer: "INT", XSD.decimal: "DOUBLE",
        XSD.float: "FLOAT", XSD.double: "DOUBLE", XSD.boolean: "BOOLEAN", XSD.date: "DATE",
        XSD.dateTime: "DATETIME", XSD.nonPositiveInteger: "INT", XSD.negativeInteger: "INT",
        XSD.long: "BIGINT", XSD.int: "INT", XSD.short: "SMALLINT", XSD.byte: "TINYINT",
        XSD.nonNegativeInteger: "INT", XSD.unsignedLong: "BIGINT UNSIGNED",
        XSD.unsignedInt: "INT UNSIGNED", XSD.unsignedShort: "SMALL INT UNSIGNED",
        XSD.unsignedByte: "TINYINT UNSIGNED", XSD.positiveInteger: "INT UNSIGNED",
        "http://www.w3.org/2001/xmlschema#gyearmonth": "DATE",
        "none": "VARCHAR(511)"
    }

    key_list = list(type_xsd2mysql_dict.keys())
    for xsd_type in key_list:
        type_xsd2mysql_dict[str(xsd_type).lower()] = type_xsd2mysql_dict[xsd_type]

    return type_xsd2mysql_dict


def filter_str(s):
    res = ""
    for c in s:
        if c.isalnum():
            res = res + c
        else:
            res = res + "_"
    return res


# 生成数据库
DATABASE_NAME = "main"
RDF_FILE_NAME = "iofiles/valid_infobox.nt"
# RDF_FILE_NAME = "tmp2.nt"
PRED_TEMPLATE = "<http://dbpedia.org/property/wikiPageUsesTemplate>"

g = rdflib.Graph()
g.parse(RDF_FILE_NAME, format="nt")

g_size = len(g)
# print(g_size)

ib_cnt = {}
subj2ib = {}
for s, p, o in g:
    if not str(s) in ib_cnt:
        ib_cnt[str(s)] = 0
    if str(p) != "http://dbpedia.org/property/wikiPageUsesTemplate":
        continue
    if str(o).startswith("http://dbpedia.org/resource/Template:infobox_"):
        ib_cnt[str(s)] += 1
        subj2ib[str(s)] = str(o).replace("http://dbpedia.org/resource/Template:infobox_", "")
        # print("{} : {}".format(str(s), ib_cnt[str(s)]))
# print(ib_cnt)

VALID_FILE_NAME = "iofiles/valid_infobox.nt"
ng = Graph()
for s, p, o in g:
    if ib_cnt[str(s)] == 1:
        ng.add((s, p, o))
print(len(ng))

type_xsd2mysql_dict = get_type_xsd2mysql_dict()
# print(type_xsd2mysql_dict)

subj_set = set()
for s in ng.subjects():
    subj_set.add(s)

# %%

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345678"
)

# print(db)

cursor = conn.cursor(buffered=True)
try:
    cursor.execute("DROP DATABASE {}".format(DATABASE_NAME))
except mysql.connector.errors.DatabaseError:
    pass

cursor.execute("CREATE DATABASE IF NOT EXISTS " + DATABASE_NAME)
cursor.execute("USE {}".format(DATABASE_NAME))

print(len(subj_set))
subj_cnt = 0
for subj in subj_set:
    subj_cnt += 1
    if subj_cnt % 200 == 0:
        print("\nInserted subjects:", subj_cnt)
        print("Completed: %.1f%%", subj_cnt / len(subj_set))

    try:
        table_name = "IB_" + filter_str(subj2ib[str(subj)])
        CMD_CREATE_TABLE = "CREATE TABLE IF NOT EXISTS {} (" \
                           "_subject_id_ INT AUTO_INCREMENT PRIMARY KEY, " \
                           "_subject_ VARCHAR(511) NOT NULL)".format(table_name)
        # print(CMD_CREATE_TABLE)
        cursor.execute(CMD_CREATE_TABLE)
        subject_name = str(subj)
        CMD_SELECT_ROW = "SELECT * FROM {} WHERE _subject_ = '{}'".format(table_name, str(subject_name))
        # print(CMD_SELECT_ROW)
        cursor.execute(CMD_SELECT_ROW)
        result = cursor.fetchall()
        if len(result) == 0:
            cursor.execute("INSERT INTO {} ({}) VALUES ('{}')".format(table_name, "_subject_", str(subject_name)))
            conn.commit()
            cursor.execute("SELECT LAST_INSERT_ID()")
            subj_id = cursor.fetchone()[0]
        else:
            subj_id = result[0][0]
    except mysql.connector.Error as err:
        # print(err)
        continue

    for s, p, o in ng.triples((subj, None, None)):
        try:
            try:
                # print(s,  p, o, o.datatype)
                sql_datatype = type_xsd2mysql_dict[str(o.datatype).lower()]
            except AttributeError or KeyError:
                sql_datatype = "VARCHAR(511)"
            column_name = "P_" + str(p).replace("http://dbpedia.org/property/", "")
            column_name = filter_str(column_name)
            CMD_ADD_COLUMN = "ALTER TABLE {} ADD COLUMN {} {}".format(table_name, column_name, sql_datatype)
            # print(CMD_ADD_COLUMN)
            try:
                cursor.execute(CMD_ADD_COLUMN)
                cursor.fetchall()
            except mysql.connector.Error as err:
                # print("INNER:", err)
                pass

            value = o.toPython()
            if sql_datatype.startswith("VARCHAR"):
                value = value.replace("'", "")
                value = value.replace('"', '')
                value = "'{}'".format(value)
                # print("value:", value)
            elif sql_datatype == "DATE":
                value = "'{}'".format(value)
            try:
                CMD_UPDATE = "UPDATE {} SET {} = {} WHERE _subject_id_ = {}" \
                    .format(table_name, column_name, value, subj_id)
                # print(CMD_UPDATE)
                cursor.execute(CMD_UPDATE)
                conn.commit()
            except mysql.connector.Error as err:
                pass
                # print("MYSQL ERR:", err)

        except:
            pass
            # print("Something Wrong")
