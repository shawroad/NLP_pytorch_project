"""

@file  : my_data_deal.py

@author: xiaolu

@time  : 2020-06-03

"""
import glob


def load_data(p, Data_tup_scene):
    with open(p, 'r', encoding='utf8') as f:
        res_data = []
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            res_data.append(line)

    s = p.split('/')[-1]
    label = -1
    for dts in Data_tup_scene:
        if dts[0] == s:
            label = dts[1]
    if label == -1:
        print("标签错误")
        print(p)
        exit()
    res_label = [label] * len(res_data)

    return res_data, res_label


if __name__ == '__main__':
    Data_tup_scene = [('control.data', 0),
                      ('app_control.data', 1),
                      ('video.data', 2),
                      ('weather.data', 3),
                      ('channel_switch.data', 4),
                      ('image_interactive.data', 5),
                      ('math_operation.data', 6),
                      ('disport.data', 7),
                      ('time_query.data', 8),
                      ('baike.data', 9),
                      ('info_news.data', 10),
                      ('info_stock&fund.data', 11),
                      ('translate.data', 12),
                      ('converter.data', 13),
                      ('karaoke.data', 14),
                      ('words.data', 15),
                      ('audio.data', 16),
                      ('map_server.data', 17),
                      ('ticket_server.data', 18),
                      ('order_server.data', 19),
                      ('third_query.data', 20),
                      ('multi_dialogue.data', 21),
                      ('unknown_server.data', 22),
                      ('search.data', 23),
                      ('couplet.data', 24),
                      ('poetry.data', 25),
                      ('relation_dialogue.data', 26),
                      ('album.data', 27),
                      ('help.data', 28),
                      ('reminder.data', 29)
                      ]
    # path = './TCL_Corpus/'
    path = './TCL_Corpus_test/'
    base_file = glob.glob(path + '*')

    data_path = []
    for bf in base_file:
        p = glob.glob(bf + '/*')
        data_path.append(p[0])

    total_data = []
    total_label = []
    for p in data_path:
        temp_data, temp_label = load_data(p, Data_tup_scene)
        total_data.append(temp_data)
        total_label.append(temp_label)

    # max_len = 0
    # for d in total_data:
    #     for i in d:
    #         if len(i) > max_len:
    #             max_len = len(i)

    write_data = []
    for d, l in zip(total_data, total_label):
        for d_i, l_i in zip(d, l):
            temp = str(l_i) + '	' + d_i
            write_data.append(temp)

    write_data = list(set(write_data))   # 对数据去重

    # # train 100000  dev 50000  test 50000
    # dev_data = write_data[:50000]
    # train_data = write_data[50000:]
    #
    # # print(len(train_data))   # 101231
    # # print(len(dev_data))   # 50000
    # # print(len(test_data))   # 50000
    # with open('./tcl/train.tsv', 'w') as f:
    #     f.write('\n'.join(train_data))
    #
    # with open('./tcl/train.tsv', 'w') as f:
    #     f.write('\n'.join(dev_data))

    print(len(write_data))
    with open('./tcl/test.tsv', 'w') as f:
        f.write('\n'.join(write_data))





