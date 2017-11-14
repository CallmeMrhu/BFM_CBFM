# user / basket / target item / isAdopt(0,1)

import random
import pandas as pd


# 对数据按照时间、user进行排序
def sortData(path, dataset):
    filePath = path + '/' + dataset + '.txt'
    file_df = pd.read_csv(filePath, sep=';', names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    file_df = file_df.loc[:, ('a', 'b', 'f')]
    file_df.sort_values(['a', 'b'], inplace=True)
    file_df.reset_index(drop=True, inplace=True)
    print("the number of row:", len(file_df))
    print("data has been sorted!")
    print(' ')
    return file_df

# user 、 item 的字典
def createDict(sortdata):
    user_dict = {}
    item_dict = {}
    user_list = list(sortdata['b'].drop_duplicates())
    item_list = list(sortdata['f'].drop_duplicates())
    user_index = 0
    for user in user_list:
        user_dict[user] = user_index
        user_index += 1
    item_index = 0
    for item in item_list:
        item_dict[item] = item_index
        item_index += 1
    return user_dict, item_dict


# user购买的所有item
def createUserToItem(user_dict, item_length, sortdata):
    user_items = {}

    for row in sortdata.itertuples():
        user_id = user_dict[row.b]
        item_id = item_dict[row.f]

        if (user_id not in user_items.keys()):
            user_items[user_id] = []
            user_items[user_id].append(item_id)
        else:
            user_items[user_id].append(item_id)

    return user_items


# user的baskets
def createBasket(sortdata, user_dict, item_dict):
    all_baskets = {}
    single_basket = []
    init_user = sortdata.loc[0, 'b']
    user_id = user_dict[init_user]
    for row in sortdata.itertuples():
        if (init_user != row.b):
            # 如果该购物篮中只有一个item，则舍弃
            if len(single_basket) >= 2:
                # 判断是否已经存在购物篮,不存在则初始化该篮子
                if user_id not in all_baskets.keys():
                    all_baskets[user_id] = []
                all_baskets[user_id].append(single_basket)

            init_user = row.b
            user_id = user_dict[init_user]
            single_basket = []

        item_id = item_dict[row.f]
        single_basket.append(item_id)

    return all_baskets


# 生成libFM的数据格式
# type为data type：train、val、test
def createTextFormat(baskets, user_length, item_length, user_items, path, train_dataset, type):
    N = user_length - 1  # user
    M = N + item_length  # the target item
    Q = M + item_length  # basket item

    # 对于每个用户，生成m*len(basket)个正样本，m为篮子个数
    # 对训练数据，每生成一个正样本，产生两个负样本

    train_TextFormat = open(path + '/' + train_dataset + '_textFormat.csv', 'a')

    for user_id in baskets.keys():
        basket_list = list(baskets[user_id])
        for basket in basket_list:
            basket_lenth = len(basket)
            # user_content = str(user_id) + ':1 '
            user_content = str(user_id) + ','
            # select target item
            for index in range(basket_lenth):
                # target_content = str(N + basket[index]) + ':1 '
                target_content = str(N + basket[index]) + ','
                basket_content = ''
                # collect basket item
                for i in range(basket_lenth):
                    if i != index:
                        # basket_content = basket_content + str(M + basket[i]) + ':1 '
                        if index !=(basket_lenth-1) and i == (basket_lenth-1):
                            basket_content = basket_content + str(M + basket[i])
                        elif index ==(basket_lenth-1) and i == (basket_lenth-2):
                            basket_content = basket_content + str(M + basket[i])
                        else:
                            basket_content = basket_content + str(M + basket[i]) + ','
                    else:
                        i += 1
                # positive class
                # text_content_positive = '1 ' + user_content + target_content + basket_content
                text_content_positive = '1,' + user_content + target_content + basket_content
                train_TextFormat.write(text_content_positive + '\n')
                # 如果是训练数据集，则每一条数据产生相同规格的负采样数据两条
                if (type == 'train'):
                    # do negtive sampling
                    # each positive item has two negtive items
                    for i in range(2):
                        item_num = 0
                        negtive_content = ''
                        basket_content = ''
                        while item_num < basket_lenth:
                            item_id = random.randint(0, item_length - 1)
                            if item_id not in user_items[user_id]:
                                if item_num == 0:
                                    # negtive_content = str(N + item_id) + ':1 '
                                    negtive_content = str(N + item_id) + ','
                                    item_num += 1
                                else:
                                    # basket_content = basket_content + str(M + item_id) + ':1 '
                                    # item_num += 1
                                    item_num += 1
                                    if item_num == basket_lenth:
                                        basket_content = basket_content + str(M + item_id)
                                    else:
                                        basket_content = basket_content + str(M + item_id) + ','
                        # text_content_negative = '-1 ' + user_content + negtive_content + basket_content
                        text_content_negative = '-1,' + user_content + negtive_content + basket_content
                        train_TextFormat.write(text_content_negative + '\n')

    return


if __name__ == '__main__':
    path = 'TaFeng'
    all_dataset = 'dataset'
    train_dataset = 'train'
    validation_dataset = 'validation'
    test_dataset = 'test'

    sortdata_all = sortData(path, all_dataset)
    user_dict, item_dict = createDict(sortdata_all)
    user_length = len(user_dict)
    item_length = len(item_dict)
    print('the number of user:', user_length)
    print('the number of item:', item_length)
    print(' ')

    sortdata_train = sortData(path, train_dataset)
    sortdata_validation = sortData(path, validation_dataset)
    sortdata_test = sortData(path, test_dataset)
    # users have buy them
    user_items = createUserToItem(user_dict, item_length, sortdata_train)

    baskets_train = createBasket(sortdata_train, user_dict, item_dict)
    print("baskets_train has been created!")
    baskets_validation = createBasket(sortdata_validation, user_dict, item_dict)
    print("baskets_validation has been created!")
    baskets_test = createBasket(sortdata_test, user_dict, item_dict)
    print("baskets_test has been created!")
    print(' ')

    createTextFormat(baskets_train, user_length, item_length, user_items, path, train_dataset, 'train')
    print('create train TextFormat finished')
    createTextFormat(baskets_validation, user_length, item_length, user_items, path, validation_dataset, 'validation')
    print('create validation TextFormat finished')
    createTextFormat(baskets_test, user_length, item_length, user_items, path, test_dataset, 'test')
    print('create test TextFormat finished')
    print(' ')
