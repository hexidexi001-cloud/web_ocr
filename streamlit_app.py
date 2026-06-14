import streamlit as st
import easyocr
import pymorphy3
from PIL import Image
import numpy as np

# ============================================
# Инициализация моделей (кэшируем для скорости)
# ============================================

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ru'], gpu=False)

@st.cache_resource
def load_morph():
    return pymorphy3.MorphAnalyzer()

reader = load_ocr()
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
                results = reader.readtext(image_np, paragraph=True)

                # Защита от неожиданной структуры результата
                recognized_text = " ".join([
                    r[1] for r in results if len(r) > 1
                ])
            except Exception as e:
                st.error(f"❌ Ошибка при распознавании текста: {e}")
                st.stop()

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
