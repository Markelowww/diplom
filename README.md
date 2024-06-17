Исследование эффективности архитектур глубокого обучения в диагностике глазных заболеваний

Классы заболеваний в датасете(OCTDL и OCTID)
    AMD
    CSR
    DME
    DR
    ERM
    MH
    NO
Всего 2017 снимков

Иерархия
```
├── dataset
    ├── train
        ├── class1
            ├── image1.jpg
            ├── ...
        ├── class2
        ├── class3
        ├── ...
    ├── val
    ├── test
```
после того как загрузил датасет нужно сделать preprocessing.py
```
python preprocessing.py
```
Стандартные настройки:
```
--dataset_folder', type=str, default='./OCTDL_folder', help='path to dataset folder')
--output_folder', type=str, default='./dataset', help='path to output folder')
--crop_ratio', type=int, default=1, help='central crop ratio of image')
--image_dim', type=int, default=512, help='final dimensions of image')
--val_ratio', type=float, default=0.1, help='validation size')
--test_ratio', type=float, default=0.2, help='test size')
--padding', type=bool, default=False, help='padding to square')
```