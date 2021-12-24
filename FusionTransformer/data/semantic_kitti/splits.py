# official split defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml

class regular(object):
    train = [
        '00',
        '02',
        '03',
        '04',
        '05',
        '06',
        '09',
        '10',
    ]

    val = [
        '07',
        '01'
        # '04'
    ]

    test = [
        '08'
        # '04'
    ]

    # not used
    hidden_test = [
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
    ]


class debug(object):
    train = [
        '07',
        '01',
    ]

    val = [
        '08',
        #  '01'
    ]

    test = [
        '08',
        # '01'

    ]


class kfolds(object):
    folds = {
    0: {"train": [ "04", "05",  "06", "07",  "08",  "09", "10"], "val": ["00", "01", "02", "03"]},
    1: {"train": [ "00", "01",  "02", "03"  "08",  "09", "10"], "val": ["04", "05", "06", "07"]},
    2: {"train": [ "00", "01",  "02", "03", "04", "05", "06",  "07"], "val": [ "08",  "09", "10"]}
    }
    def get_seqs(self, fold, name):
        return self.folds[fold][name]