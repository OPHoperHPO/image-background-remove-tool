# <p align="center"> ✂️ CarveKit ✂️  </p>

<p align="center"> <img src="/docs/imgs/logo.png"> </p>

<p align="center">
<img src="https://github.githubassets.com/favicons/favicon-success.svg"> <a src="https://github.com/OPHoperHPO/image-background-remove-tool/actions">
<img src="https://github.com/OPHoperHPO/image-background-remove-tool/workflows/Test%20release%20version/badge.svg?branch=master"> <a src="https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb">
<a href="https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb">
<img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>

</p>

**********************************************************************
<p align="center"> <img align="center" width="512" height="288" src="/docs/imgs/compare/readme.jpg"> </p>

> Изображения с более высоким разрешением с примером выше можно увидеть в директории docs/imgs/compare/ и docs/imgs/input folders.

#### 📙 README Language
[Russian](/docs/readme/ru.md)
[English](/README.md)

## 📄 О проекте:  
Автоматизированное высококачественное удаление фона с изображения с использованием нейронных сетей


## 🎆 Особенности:  
- Высокое качество выходного изображения
- Работает в автономном режиме
- Пакетная обработка изображений
- Поддержка NVIDIA CUDA и процессорной обработки
- Поддержка FP16: быстрая обработка с низким потреблением памяти
- Легкое взаимодействие и запуск
- 100% совместимое с remove.bg API FastAPI HTTP API
- Удаляет фон с волос
- Автоматический выбор лучшего метода для изображения пользователя
- Простая интеграция с вашим кодом
- Модели размещены на [HuggingFace](https://huggingface.co/Carve)

## ⛱ Попробуйте сами на [Google Colab](https://colab.research.google.com/github/OPHoperHPO/image-background-remove-tool/blob/master/docs/other/carvekit_try.ipynb) 
## ⛓️ Как это работает?

1. Пользователь выбирает картинку или папку с картинками для обработки
2. Происходит предобработка фотографии для обеспечения лучшего качества выходного изображения
3. С помощью технологии машинного обучения убирается фон у изображения
4. Происходит постобработка изображения для улучшения качества обработанного изображения

## 🎓 Implemented Neural Networks:
| Нейронные сети |               Целевая область                |             Точность             |
|:--------------:|:--------------------------------------------:|:--------------------------------:|
| **Tracer-B7**  |  **Общий** (objects, people, animals, etc)   | **90%** (mean F1-Score, DUTS-TE) |
|    U^2-net     | **Волосы** (hairs, people, animals, objects) |   80% (mean F1-Score, DUTS-TE)   |
|     BASNet     |         **Общий** (people, objects)          |   80% (mean F1-Score, DUTS-TE)   |
|   DeepLabV3    |          People, Animals, Cars, etc          |  67.4% (mean IoU, COCO val2017)  |

### Recommended parameters for different models
| Нейронные сети | Размер маски сегментации | Параметры Trimap (расширение, эрозия) |
|:--------------:|:------------------------:|:-------------------------------------:|
|  `tracer_b7`   |           640            |                (30, 5)                |
|    `u2net`     |           320            |                (30, 5)                |
|    `basnet`    |           320            |                (30, 5)                |
|  `deeplabv3`   |           1024           |               (40, 20)                |

> ### Notes: 
> 1. Окончательное качество может зависеть от разрешения вашего изображения, типа сцены или объекта.
> 2. Используйте U2-Net для волос и Tracer-B7 для общих изображений и правильных параметров. \
> Это очень важно для конечного качества! Примеры изображений были получены с использованием постобработки U2-Net и FBA.

## 🖼️ Image pre-processing and post-processing methods:
### 🔍 Preprocessing methods:
* `none` - No preprocessing methods used.
* [`autoscene`](https://huggingface.co/Carve/scene_classifier/) - Автоматически определяет тип сцены с помощью классификатора и применяет соответствующую модель. (По умолчанию)
* `auto` - Выполняет глубокий анализ изображения и более точно определяет лучший метод удаления фона. Использует классификатор объектов и классификатор сцены вместе.
> ### Notes: 
> 1. `AutoScene` и `auto` могут переопределить модель и параметры, указанные пользователем, без уведомления.
> Итак, если вы хотите использовать конкретную модель, сделать все постоянными и т. д., вам следует сначала отключить методы автоматической предварительной обработки!
> 2. На данный момент для метода `auto` выбираются универсальные модели для некоторых конкретных доменов, так как добавленных моделей в настоящее время недостаточно для такого количества типов сцен.
> В будущем, когда будет добавлено некоторое разнообразие моделей, автоподбор будет переписан в лучшую сторону.

### ✂ Методы постобработки:
* `none` - методы постобработки не используются
* `fba` - Этот алгоритм улучшает границы изображения при удалении фона с изображений с волосами и т.д. с помощью нейронной сети FBA Matting.
* `cascade_fba` (default) - Этот алгоритм уточняет маску сегментации с помощью нейронной сети CascadePSP, а затем применяет алгоритм FBA.

## 🏷 Настройка для обработки на CPU:
1. `pip install carvekit --extra-index-url https://download.pytorch.org/whl/cpu`
> Проект поддерживает версии Python от 3.8 до 3.10.4.

## 🏷 Настройка для обработки на GPU:  
1. Убедитесь, что у вас есть графический процессор NVIDIA с 8 ГБ видеопамяти.
2. Установите `CUDA Toolkit и Видеодрайвер для вашей видеокарты.`
3. `pip install carvekit --extra-index-url https://download.pytorch.org/whl/cu113`
> Проект поддерживает версии Python от 3.8 до 3.10.4.

## 🧰 Интеграция в код:  
### Если вы хотите быстрее приступить к работе без дополнительной настройки
``` python
import torch
from carvekit.api.high import HiInterface

# Check doc strings for more information
interface = HiInterface(object_type="auto",  # Can be "object" or "hairs-like" or "auto"
                        batch_size_seg=5,
                        batch_size_pre=5,
                        batch_size_matting=1,
                        batch_size_refine=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        refine_mask_size=900,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)
images_without_background = interface(['./tests/data/cat.jpg'])
cat_wo_bg = images_without_background[0]
cat_wo_bg.save('2.png')

                   
```
### Аналог метода предварительной обработки `auto` из cli
``` python
from carvekit.api.autointerface import AutoInterface
from carvekit.ml.wrap.scene_classifier import SceneClassifier
from carvekit.ml.wrap.yolov4 import SimplifiedYoloV4

scene_classifier = SceneClassifier(device="cpu", batch_size=1)
object_classifier = SimplifiedYoloV4(device="cpu", batch_size=1)

interface = AutoInterface(scene_classifier=scene_classifier,
                          object_classifier=object_classifier,
                          segmentation_batch_size=1,
                          postprocessing_batch_size=1,
                          postprocessing_image_size=2048,
                          refining_batch_size=1,
                          refining_image_size=900,
                          segmentation_device="cpu",
                          fp16=False,
                          postprocessing_device="cpu")
images_without_background = interface(['./tests/data/cat.jpg'])
cat_wo_bg = images_without_background[0]
cat_wo_bg.save('2.png')
```
### Если вы хотите провести детальную настройку
``` python
import PIL.Image

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.scene_classifier import SceneClassifier
from carvekit.ml.wrap.cascadepsp import CascadePSP
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import CasMattingMethod
from carvekit.pipelines.preprocessing import AutoScene
from carvekit.trimap.generator import TrimapGenerator

# Check doc strings for more information
seg_net = TracerUniversalB7(device='cpu',
                            batch_size=1, fp16=False)
cascade_psp = CascadePSP(device='cpu',
                         batch_size=1,
                         input_tensor_size=900,
                         fp16=False,
                         processing_accelerate_image_size=2048,
                         global_step_only=False)
fba = FBAMatting(device='cpu',
                 input_tensor_size=2048,
                 batch_size=1, fp16=False)

trimap = TrimapGenerator(prob_threshold=231, kernel_size=30, erosion_iters=5)

scene_classifier = SceneClassifier(device='cpu', batch_size=5)
preprocessing = AutoScene(scene_classifier=scene_classifier)

postprocessing = CasMattingMethod(
    refining_module=cascade_psp,
    matting_module=fba,
    trimap_generator=trimap,
    device='cpu')

interface = Interface(pre_pipe=preprocessing,
                      post_pipe=postprocessing,
                      seg_pipe=seg_net)

image = PIL.Image.open('tests/data/cat.jpg')
cat_wo_bg = interface([image])[0]
cat_wo_bg.save('2.png')     
```


## 🧰 Запустить через консоль:  
 * ```python3 -m carvekit  -i <input_path> -o <output_path> --device <device>```  
 
### Все доступные аргументы:  
````
Usage: carvekit [OPTIONS]

  Performs background removal on specified photos using console interface.

Options:
  -i ./2.jpg                   Путь до входного файла или директории  [обязателен]
  -o ./2.png                   Путь для сохранения результата обработки
  --pre autoscene              Метод предобработки
  --post cascade_fba           Метод постобработки
  --net tracer_b7              Нейронная сеть для сегментации
  --recursive                  Включение рекурсивного поиска изображений в папке
  --batch_size 10              Размер пакета изображений, загруженных в ОЗУ 
  --batch_size_pre 5           Размер пакета для списка изображений, которые будут обрабатываться
                               методом предварительной обработки
  --batch_size_seg 5           Размер пакета изображений для обработки с помощью
                               сегментации

  --batch_size_mat 1           Размер пакета изображений для обработки с помощью
                               матирования

  --batch_size_refine 1        Размер пакета для списка изображений, которые будут обрабатываться уточняющей сетью

  --seg_mask_size 640          Размер исходного изображения для сегментирующей
                               нейронной сети

  --matting_mask_size 2048     Размер исходного изображения для матирующей
                               нейронной сети
  --refine_mask_size 900       Размер входного изображения для уточняющей нейронной сети.
  --trimap_dilation 30         Размер радиуса смещения от маски объекта в пикселях при 
                               формировании неизвестной области
                               
  --trimap_erosion 5           Количество итераций эрозии, которым будет подвергаться маска 
                               объекта перед формированием неизвестной области.
                               
  --trimap_prob_threshold 231  Порог вероятности, при котором будут применяться
                               операции prob_filter и prob_as_unknown_area

  --device cpu                 Устройство обработки.
  
  --fp16                       Включает обработку со смешанной точностью. 
                               Используйте только с CUDA. Поддержка процессора является экспериментальной!
                               
  --help                       Показать это сообщение и выйти.

````
## 📦 Запустить фреймворк / FastAPI HTTP API сервер с помощью Docker:

Использование API через Docker — это **быстрый** и эффективный способ получить работающий API.\
> Наши образы Docker доступны на [Docker Hub](https:hub.docker.comranodevcarvekit). \
> Теги версий совпадают с релизами проекта с суффиксами `-cpu` и `-cuda` для версий CPU и CUDA соответственно.


<p align="center"> 
<img src="/docs/imgs/screenshot/frontend.png"> 
<img src="/docs/imgs/screenshot/docs_fastapi.png"> 
</p>

>### Важная информация:
>1. Образ Docker имеет фронтенд по умолчанию по адресу `/` и документацию к API по адресу `/docs`.
>2. Аутентификация **включена** по умолчанию. \
> **Ключи доступа сбрасываются** при каждом перезапуске контейнера, если не установлены специальные переменные окружения. \
См. `docker-compose.<device>.yml` для более подробной информации. \
> **Вы можете посмотреть свои ключи доступа в логах докера.**
> 
>3. Примеры работы с API.\
> См. `docs/code_examples/python` для уточнения деталей
### 🔨 Создать и запустить контейнер:
1. Установите `docker-compose`
2. Запустите `docker-compose -f docker-compose.cpu.yml up -d`  # для обработки на ЦП
3. Запустите `docker-compose -f docker-compose.cuda.yml up -d`  # для обработки на GPU
> Также вы можете монтировать папки с вашего компьютера в docker container
> и использовать интерфейс командной строки внутри контейнера докера для обработки
> файлов в этой папке.

> Создание docker образа в Windows официально не поддерживается. Однако вы можете попробовать использовать WSL2 или «Linux container mode» в Docker Desktop.
## ☑️ Тестирование

### ☑️ Тестирование с локальным окружением
1. `pip install -r requirements_test.txt`
2. `pytest`

### ☑️ Тестирование с Docker
1. Запустите `docker-compose -f docker-compose.cpu.yml run carvekit_api pytest`  # для тестирования на ЦП
2. Run `docker-compose -f docker-compose.cuda.yml run carvekit_api pytest`  # для тестирования на GPU


## 👪 Credits: [Больше информации](/docs/CREDITS.md)

## 💵 Поддержать развитие проекта
  Вы можете поблагодарить меня за разработку этого проекта и угостить меня чашечкой кофе ☕

| Blockchain |            Cryptocurrency           |          Network          |                                              Wallet                                             |
|:----------:|:-----------------------------------:|:-------------------------:|:-----------------------------------------------------------------------------------------------:|
|  Ethereum  | ETH / USDT / USDC / BNB / Dogecoin  |          Mainnet          |                            0x7Ab1B8015020242D2a9bC48F09b2F34b994bc2F8                           |
|  Ethereum  |  ETH / USDT / USDC / BNB / Dogecoin | BSC (Binance Smart Chain) |                            0x7Ab1B8015020242D2a9bC48F09b2F34b994bc2F8                           |
|   Bitcoin  |                 BTC                 |             -             |                            bc1qmf4qedujhhvcsg8kxpg5zzc2s3jvqssmu7mmhq                           |
|    ZCash   |                 ZEC                 |             -             |                               t1d7b9WxdboGFrcVVHG2ZuwWBgWEKhNUbtm                               |
|    Tron    |                 TRX                 |             -             |                                TH12CADSqSTcNZPvG77GVmYKAe4nrrJB5X                               |
|   Monero   |                 XMR                 |          Mainnet          | 48w2pDYgPtPenwqgnNneEUC9Qt1EE6eD5MucLvU3FGpY3SABudDa4ce5bT1t32oBwchysRCUimCkZVsD1HQRBbxVLF9GTh3 |
|     TON    |                 TON                 |             -             |                         EQCznqTdfOKI3L06QX-3Q802tBL0ecSWIKfkSjU-qsoy0CWE                        |

## 📧 __Обратная связь__
Буду рад отзывам о проекте и предложениям об интеграции.

По всем вопросам писать: [farvard34@gmail.com](mailto://farvard34@gmail.com)
