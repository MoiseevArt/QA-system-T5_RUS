from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import config
import time


bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


async def error(chat_id):
    await bot.send_message(chat_id=chat_id, text='Что-то пошло не так...')


async def generate_answer(chat_id, question, context):
    try:
        input_text = f"question: {question} context: {context}"
        input_ids = config.tokenizer.encode(input_text, return_tensors='pt')

        outputs = config.model.generate(input_ids)
        answer = config.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer

    except:
        await error(chat_id=chat_id)


@dp.message_handler(commands=['start'])
async def welcome(message: types.Message):
    try:

        if config.first:
            await generate_answer(message.chat.id, "init", "init")
            config.first = False

        await bot.send_chat_action(chat_id=message.chat.id, action='typing')

        await bot.send_message(chat_id=message.chat.id,
                               text=f"Добро пожаловать, {message.from_user.first_name}!🪄\n"
                                    f"Этот бот помогает найти ответ на вопрос, используя указанный вами текст 🔍\n"
                                    f"Для начала введите контекст (текст), в котором содержится ответ на ваш вопрос:")

    except:
        await error(chat_id=message.chat.id)


@dp.message_handler(content_types='text')
async def mgs(message: types.Message):
    try:
        if message.chat.id not in config.users_control:

            config.users_control[message.chat.id] = message.text

            await bot.send_chat_action(chat_id=message.chat.id, action='typing')
            await bot.send_message(chat_id=message.chat.id, text='Отлично, теперь отправьте ваш вопрос: ')

        else:
            await bot.send_chat_action(chat_id=message.chat.id, action='typing')
            await bot.send_message(chat_id=message.chat.id,
                                   text=f"Ответ:\n{await generate_answer(message.chat.id, message.text, config.users_control[message.chat.id])}")

            del config.users_control[message.chat.id]

            await bot.send_message(chat_id=message.chat.id, text=f"Введите следующий контекст:")

    except:
        await error(message.chat.id)


@dp.message_handler(content_types=['voice', 'video', 'audio', 'document', 'sticker', 'photo', 'location',
                                   'contact', 'poll', 'photo', 'video_note'])
async def other(message: types.Message):
    try:
        time.sleep(0.5)
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)

    except:
        await error(message.chat.id)


if __name__ == '__main__':
    try:
        executor.start_polling(dp, skip_updates=False)

    except Exception as e:
        time.sleep(3)
