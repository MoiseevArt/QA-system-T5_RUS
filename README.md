# QA system T5 [RUS]

### Описание:
Данный проект представляет собой русскоязычную модель для задач "question-answer" (QA), которая генерирует ответы на вопросы, исходя из контекста. В основе модели лежит дообученная версия **ruT5-base** — трансформера T5, адаптированного для русского языка. Модель была дополнительно обучена (fine-tuned) на русскоязычном датасете **SberQUAD**, что позволяет ей эффективно решать задачи извлечения информации из текста. Сама модель доступна на: https://huggingface.co/oOundefinedOo/QA-system-T5_RUS
___
### Детали:
- **Датасет**: SberQUAD — аналог английского SQuAD, но для русского языка. Содержит вопросы и ответы, основанные на реальных текстах.
- **Точность**: После дообучения на SberQUAD модель демонстрирует высокую точность в генерации ответов на заданные вопросы. 
  - F1-мера: ~ 0.85
  - EM (Exact Match): ~ 0.6
- **Области применения**: Автоматизация чат-ботов, интеллектуальные системы поиска, образовательные платформы и другие проекты, требующие обработки естественного языка.
___
### Использование:
- **`training.py`** — основной файл для запуска процесса файнтюнинга модели.
- **`example.py`** — пример использования модели.
- **`metrics.py`** — содержит показательный инференс модели на тестовом наборе данных, а также код для вычисления метрик F1 и EM.
___
### Установка зависимостей:
Установите зависимости из файла `requirements.txt` с помощью pip:
```bash
pip install -r requirements.txt
```
