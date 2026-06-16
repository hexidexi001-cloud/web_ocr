import streamlit as st
import easyocr
import pymorphy3
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ============================================
# Инициализация моделей (кэшируем для скорости)
# ============================================

@st.cache_resource
def load_ocr():
    reader = easyocr.Reader(['ru'], gpu=False)
    hf_model_repo = "AnatoliiPa/trocr_model"

    processor = TrOCRProcessor.from_pretrained(hf_model_repo)
    model = VisionEncoderDecoderModel.from_pretrained(hf_model_repo)

    model.to("cpu")

    return reader, processor, model

reader, processor, model = load_ocr()

@st.cache_resource
def load_morph():
    return pymorphy3.MorphAnalyzer()

morph = load_morph()

# ============================================
# Функции проверки грамотности
# ============================================

def check_spelling(word, morph):
    """Проверка орфографии через pymorphy3"""
    clean = word.strip(".,!?;:\"'()-–—")
    if not clean:
        return None

    if clean.isdigit():
        return None

    parsed = morph.parse(clean.lower())
    best = parsed[0]

    if best.score < 0.01 and len(clean) > 2:
        return {
            "word": word,
            "type": "Орфографическая",
            "suggestion": best.normal_form,
            "confidence": 1 - best.score
        }
    return None


def check_grammar_simple(words, morph):
    """Простая проверка согласования прилагательного и существительного"""
    errors = []
    for i in range(len(words) - 1):
        w1 = words[i].strip(".,!?;:\"'()-–—").lower()
        w2 = words[i + 1].strip(".,!?;:\"'()-–—").lower()

        if not w1 or not w2:
            continue

        p1 = morph.parse(w1)[0]
        p2 = morph.parse(w2)[0]

        if ('ADJF' in p1.tag and 'NOUN' in p2.tag):
            gender1 = p1.tag.gender
            gender2 = p2.tag.gender
            if gender1 and gender2 and gender1 != gender2:
                errors.append({
                    "word": f"{words[i]} {words[i+1]}",
                    "type": "Грамматическая",
                    "suggestion": f"Несогласование по роду: {gender1} ≠ {gender2}",
                    "confidence": 0.8
                })

            number1 = p1.tag.number
            number2 = p2.tag.number
            if number1 and number2 and number1 != number2:
                errors.append({
                    "word": f"{words[i]} {words[i+1]}",
                    "type": "Грамматическая",
                    "suggestion": f"Несогласование по числу: {number1} ≠ {number2}",
                    "confidence": 0.8
                })
    return errors


def analyze_text(text, morph):
    """Полный анализ текста"""
    words = text.split()
    errors = []

    for word in words:
        error = check_spelling(word, morph)
        if error:
            errors.append(error)

    grammar_errors = check_grammar_simple(words, morph)
    errors.extend(grammar_errors)

    return errors, words


def calculate_metrics(errors, words):
    """Вычисление метрик грамотности"""
    total_words = len(words)
    total_errors = len(errors)

    if total_words == 0:
        return 0, 0, 100

    errors_per_100 = (total_errors / total_words) * 100
    error_rate = (total_errors / total_words) * 100
    score = max(0, 100 - errors_per_100 * 5)

    return errors_per_100, error_rate, score


def prepare_image(uploaded_file):
    """
    Открывает изображение, конвертирует в RGB
    и ограничивает размер для безопасной обработки
    """
    image = Image.open(uploaded_file)

    # Конвертируем любой режим (RGBA, L, P, CMYK и др.) в RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Ограничиваем размер чтобы избежать OOM и таймаутов
    MAX_SIZE = (1800, 1800)
    image.thumbnail(MAX_SIZE, Image.LANCZOS)

    return image


# ============================================
# Интерфейс Streamlit
# ============================================

st.title("📝 Оценка грамотности рукописного текста")
st.markdown("Загрузите фотографию рукописного текста для автоматической проверки")

# Боковая панель
st.sidebar.header("⚙️ Настройки")
mode = st.sidebar.radio(
    "Режим ввода:",
    ["📷 Загрузить изображение", "⌨️ Ввести текст вручную"]
)

# ============================================
# Режим 1: Загрузка изображения
# ============================================

if mode == "📷 Загрузить изображение":
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        # Подготовка изображения с обработкой ошибок
        try:
            image = prepare_image(uploaded_file)
        except Exception as e:
            st.error(f"❌ Не удалось открыть изображение: {e}")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Загруженное изображение")
            st.image(image, use_container_width=True)

        # Распознавание текста с обработкой ошибок
        with st.spinner("🔄 Распознавание текста..."):
            try:
                image_np = np.array(image)
                ocr_results = reader.readtext(image_np)

                if len(ocr_results) == 0:
                    st.warning("Текст на изображении не обнаружен.")
                else:
                    # 2. Сортируем рамки сверху вниз по координате y верхнего левого угла (box[0][0][1])
                    ocr_results = sorted(ocr_results, key=lambda x: x[0][0][1])            
                    full_page_text = []
                    grouped_lines = []
                    if len(ocr_results) > 0:
                        current_line = [ocr_results[0]]
                        
                        for result in ocr_results[1:]:
                            # Получаем координату Y текущего слова и предыдущего слова
                            prev_y = current_line[-1][0][0][1]
                            curr_y = result[0][0][1]
                            
                            # Получаем среднюю высоту рамки, чтобы задать порог
                            prev_height = current_line[-1][0][3][1] - current_line[-1][0][0][1]
                            
                            # Если разница по высоте между словами меньше половины высоты буквы — это одна строка!
                            if abs(curr_y - prev_y) < (prev_height * 0.5):
                                current_line.append(result)
                            else:
                                grouped_lines.append(current_line)
                                current_line = [result]
                        grouped_lines.append(current_line)
                    
                    # 2. Теперь внутри каждой строки сортируем слова слева направо по координате X
                    final_sorted_results = []
                    for line in grouped_lines:
                        line_sorted_by_x = sorted(line, key=lambda x: x[0][0][0])
                        final_sorted_results.extend(line_sorted_by_x)
                    
                        
                    # 3. Проходим циклом по каждой найденной строке
                    for result in final_sorted_results:
                        box = result[0] # Получаем только массив координат углов
                        
                        # Извлекаем крайние точки для прямоугольного кропа (xmin, ymin, xmax, ymax)
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        
                        xmin, xmax = int(min(x_coords)), int(max(x_coords))
                        ymin, ymax = int(min(y_coords)), int(max(y_coords))

                        # Вырезаем полоску-строку из оригинального PIL-изображения с небольшим отступом
                        line_crop = image.crop((max(0, xmin - 3), max(0, ymin - 3), xmax + 3, ymax + 3))
                        
                        # 4. Распознаем вырезанную строку с помощью вашей модели TrOCR
                        pixel_values = processor(line_crop, return_tensors="pt").pixel_values
                        
                        with torch.no_grad():
                            generated_ids = model.generate(pixel_values)
                            
                        # Декодируем токены ИИ в понятный русский текст
                        line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                        
                        # Если модель вернула список строк, берем первую
                        if isinstance(line_text, list):
                            line_text = line_text[0] if len(line_text) > 0 else ""
                            
                        full_page_text.append(line_text)

                     # Объединяем все распознанные строки через перенос строки
                    final_result_text = "\n".join(full_page_text)
                    
                    # Выводим итоговый текст в интерфейс Streamlit
                    st.text_area("Результат распознавания:", value=final_result_text, height=300)

        
            except Exception as e:
                st.warning(e)

        with col2:
            st.subheader("Распознанный текст")
            recognized_text = st.text_area(
                "Вы можете исправить ошибки OCR:",
                value=recognized_text,
                height=200
            )

        # Анализ грамотности
        if st.button("🔍 Проверить грамотность", type="primary"):
            if recognized_text.strip():
                try:
                    errors, words = analyze_text(recognized_text, morph)
                    errors_per_100, error_rate, score = calculate_metrics(errors, words)

                    # Метрики
                    st.subheader("📊 Результаты")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Слов в тексте", len(words))
                    m2.metric("Ошибок найдено", len(errors))
                    m3.metric("Ошибок на 100 слов", f"{errors_per_100:.1f}")
                    m4.metric("Оценка грамотности", f"{score:.0f}/100")

                    st.progress(score / 100)

                    if errors:
                        st.subheader("❌ Найденные ошибки")
                        for i, e in enumerate(errors, 1):
                            with st.expander(f"Ошибка {i}: «{e['word']}» — {e['type']}"):
                                st.write(f"**Тип:** {e['type']}")
                                st.write(f"**Слово/фраза:** {e['word']}")
                                st.write(f"**Пояснение:** {e['suggestion']}")
                                st.write(f"**Уверенность:** {e['confidence']:.0%}")
                    else:
                        st.success("✅ Ошибок не найдено!")

                except Exception as e:
                    st.error(f"❌ Ошибка при анализе текста: {e}")
            else:
                st.warning("⚠️ Текст не распознан. Попробуйте другое изображение.")

# ============================================
# Режим 2: Ввод текста вручную
# ============================================

else:
    st.subheader("Введите текст для проверки")
    manual_text = st.text_area(
        "Текст:",
        placeholder="Введите текст на русском языке...",
        height=200
    )

    if st.button("🔍 Проверить грамотность", type="primary"):
        if manual_text.strip():
            try:
                errors, words = analyze_text(manual_text, morph)
                errors_per_100, error_rate, score = calculate_metrics(errors, words)

                # Метрики
                st.subheader("📊 Результаты")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Слов в тексте", len(words))
                m2.metric("Ошибок найдено", len(errors))
                m3.metric("Ошибок на 100 слов", f"{errors_per_100:.1f}")
                m4.metric("Оценка грамотности", f"{score:.0f}/100")

                st.progress(score / 100)

                if errors:
                    st.subheader("❌ Найденные ошибки")
                    for i, e in enumerate(errors, 1):
                        with st.expander(f"Ошибка {i}: «{e['word']}» — {e['type']}"):
                            st.write(f"**Тип:** {e['type']}")
                            st.write(f"**Слово/фраза:** {e['word']}")
                            st.write(f"**Пояснение:** {e['suggestion']}")
                            st.write(f"**Уверенность:** {e['confidence']:.0%}")
                else:
                    st.success("✅ Ошибок не найдено!")

            except Exception as e:
                st.error(f"❌ Ошибка при анализе текста: {e}")
        else:
            st.warning("⚠️ Введите текст для проверки")

# Подвал
st.markdown("---")
