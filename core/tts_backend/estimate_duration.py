import syllables
from pypinyin import pinyin, Style
from g2p_en import G2p
from typing import Optional
import re

class AdvancedSyllableEstimator:
    def __init__(self, default_language: Optional[str] = None):
        self.g2p_en = G2p()
        self.duration_params = {
            'en': 0.225,
            'zh': 0.21,
            'ja': 0.21,
            'fr': 0.22,
            'es': 0.22,
            'ru': 0.22,
            'ko': 0.21,
            'default': 0.22,
        }
        self.supported_languages = {'en', 'zh', 'ja', 'fr', 'es', 'ru', 'ko'}
        self.language_aliases = {
            'english': 'en',
            'en': 'en',
            'en-us': 'en',
            'en-gb': 'en',
            '简体中文': 'zh',
            '繁體中文': 'zh',
            '中文': 'zh',
            'zh': 'zh',
            'zh-cn': 'zh',
            'zh-tw': 'zh',
            'zh-hans': 'zh',
            'zh-hant': 'zh',
            '日本語': 'ja',
            'ja': 'ja',
            'jp': 'ja',
            'español': 'es',
            'espanol': 'es',
            'spanish': 'es',
            'es': 'es',
            'русский': 'ru',
            'русский язык': 'ru',
            'russian': 'ru',
            'ru': 'ru',
            'français': 'fr',
            'francais': 'fr',
            'french': 'fr',
            'fr': 'fr',
            'ko': 'ko',
            'korean': 'ko',
        }
        self.lang_patterns = {
            'ja': r'[\u3040-\u309f\u30a0-\u30ff]',
            'zh': r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]',
            'ru': r'[А-Яа-яЁё]',
            'ko': r'[\uac00-\ud7af\u1100-\u11ff]',
        }
        self.lang_joiners = {'zh': '', 'ja': '', 'en': ' ', 'fr': ' ', 'es': ' ', 'ru': ' ', 'ko': ' '}
        self.vowels_map = {
            'fr': 'aeiouyàâéèêëîïôùûüÿœæ',
            'es': 'aeiouáéíóúü',
            'ru': 'аеёиоуыэюя',
        }
        self.latin_markers = {
            'fr': r'[àâçéèêëîïôùûüÿœæ]',
            'es': r'[áéíóúñ¿¡]',
        }
        self.latin_stopwords = {
            'en': {
                'the', 'and', 'you', 'that', 'this', 'with', 'for', 'have', 'are',
                'not', 'your', 'but', 'what', 'all', 'can', 'was', 'will', 'from',
                'they', 'their', 'there', 'about', 'into', 'would', 'really',
            },
            'es': {
                'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del',
                'que', 'y', 'en', 'por', 'para', 'con', 'como', 'pero', 'mas',
                'más', 'esta', 'este', 'esto', 'eso', 'hay', 'ser', 'al',
            },
            'fr': {
                'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'en',
                'pour', 'avec', 'mais', 'que', 'qui', 'dans', 'sur', 'pas', 'plus',
                'est', 'ce', 'cette', 'ces', 'au', 'aux',
            },
        }
        self.punctuation = {
            'mid': r'[，；：,;、]+', 'end': r'[。！？.!?]+', 'space': r'\s+',
            'pause': {'space': 0.15, 'default': 0.1}
        }
        self.default_language = self._normalize_language(default_language)

    def estimate_duration(self, text: str, lang: Optional[str] = None) -> float:
        return self.process_mixed_text(text, lang)['estimated_duration']

    def count_syllables(self, text: str, lang: Optional[str] = None) -> int:
        if not text.strip():
            return 0
        lang = self._normalize_language(lang) or self._detect_language(text, self.default_language)
        
        if lang == 'en':
            return self._count_english_syllables(text)
        elif lang == 'zh':
            text = re.sub(r'[^\u4e00-\u9fff]', '', text)
            return len(pinyin(text, style=Style.NORMAL))
        elif lang == 'ja':
            text = re.sub(r'[きぎしじちぢにひびぴみり][ょゅゃ]', 'X', text)
            text = re.sub(r'[っー]', '', text)
            return len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
        elif lang in ('fr', 'es', 'ru'):
            text = text.lower()
            if lang == 'fr':
                text = re.sub(r"\b([^\W\d_]+?)(?:e|es|ent)\b", r"\1", text, flags=re.UNICODE)
            return self._count_vowel_groups(text, self.vowels_map[lang])
        elif lang == 'ko':
            return len(re.findall(r'[\uac00-\ud7af]', text))
        return len(text.split())

    def _count_english_syllables(self, text: str) -> int:
        total = 0
        for word in text.strip().split():
            try:
                total += syllables.estimate(word)
            except:
                phones = self.g2p_en(word)
                total += max(1, len([p for p in phones if any(c in p for c in 'aeiou')]))
        return max(1, total)

    def _normalize_language(self, lang: Optional[str]) -> Optional[str]:
        if not lang or not isinstance(lang, str):
            return None
        normalized = lang.strip().casefold()
        if normalized in self.language_aliases:
            return self.language_aliases[normalized]
        if normalized in self.supported_languages:
            return normalized
        return None

    def _count_vowel_groups(self, text: str, vowels: str) -> int:
        words = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
        total = 0
        vowel_pattern = f"[{re.escape(vowels)}]+"
        for word in words:
            total += max(1, len(re.findall(vowel_pattern, word)))
        return max(1, total)

    def _detect_latin_language(self, text: str, fallback_language: Optional[str] = None) -> str:
        lowered = text.casefold()

        if re.search(self.latin_markers['es'], lowered):
            return 'es'
        if re.search(self.latin_markers['fr'], lowered):
            return 'fr'

        words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", lowered, re.UNICODE)
        scores = {lang: 0 for lang in ('en', 'es', 'fr')}
        for word in words:
            for lang, stopwords in self.latin_stopwords.items():
                if word in stopwords:
                    scores[lang] += 1

        if re.search(r"\b(?:l|d|qu|j|c|n|m|t|s)'[^\W\d_]+", lowered, re.UNICODE):
            scores['fr'] += 2
        if re.search(r"\b(?:al|del)\b", lowered, re.UNICODE):
            scores['es'] += 1

        best_lang = max(scores, key=scores.get)
        top_score = scores[best_lang]
        if top_score > 0 and list(scores.values()).count(top_score) == 1:
            return best_lang

        if fallback_language in {'en', 'es', 'fr'}:
            return fallback_language
        return 'en'

    def _detect_language(self, text: str, fallback_language: Optional[str] = None) -> str:
        fallback_language = self._normalize_language(fallback_language)

        for lang in ('ja', 'zh', 'ru', 'ko'):
            if re.search(self.lang_patterns[lang], text):
                return lang
        if re.search(r"[A-Za-zÀ-ÿ]", text):
            return self._detect_latin_language(text, fallback_language)
        if fallback_language:
            return fallback_language
        return 'en'

    def process_mixed_text(self, text: str, lang: Optional[str] = None) -> dict:
        if not text or not isinstance(text, str):
            return {
                'language_breakdown': {},
                'total_syllables': 0,
                'punctuation': [],
                'spaces': [],
                'estimated_duration': 0
            }
            
        fallback_language = self._normalize_language(lang) or self.default_language
        result = {'language_breakdown': {}, 'total_syllables': 0, 'punctuation': [], 'spaces': []}
        segments = re.split(f"({self.punctuation['space']}|{self.punctuation['mid']}|{self.punctuation['end']})", text)
        total_duration = 0
        
        for i, segment in enumerate(segments):
            if not segment: continue
            
            if re.match(self.punctuation['space'], segment):
                prev_lang = self._detect_language(segments[i-1], fallback_language) if i > 0 else None
                next_lang = self._detect_language(segments[i+1], fallback_language) if i < len(segments)-1 else None
                if prev_lang and next_lang and (self.lang_joiners[prev_lang] == '' or self.lang_joiners[next_lang] == ''):
                    result['spaces'].append(segment)
                    total_duration += self.punctuation['pause']['space']
            elif re.match(f"{self.punctuation['mid']}|{self.punctuation['end']}", segment):
                result['punctuation'].append(segment)
                total_duration += self.punctuation['pause']['default']
            else:
                detected_lang = self._detect_language(segment, fallback_language)
                if detected_lang:
                    syllables = self.count_syllables(segment, detected_lang)
                    if detected_lang not in result['language_breakdown']:
                        result['language_breakdown'][detected_lang] = {'syllables': 0, 'text': ''}
                    result['language_breakdown'][detected_lang]['syllables'] += syllables
                    result['language_breakdown'][detected_lang]['text'] += (self.lang_joiners[detected_lang] + segment 
                        if result['language_breakdown'][detected_lang]['text'] else segment)
                    result['total_syllables'] += syllables
                    total_duration += syllables * self.duration_params.get(detected_lang, self.duration_params['default'])
        
        result['estimated_duration'] = total_duration
        
        return result
    
def init_estimator(default_language: Optional[str] = None):
    return AdvancedSyllableEstimator(default_language=default_language)

def estimate_duration(text: str, estimator: AdvancedSyllableEstimator, lang: Optional[str] = None):
    if not text or not isinstance(text, str):
        return 0
    return estimator.process_mixed_text(text, lang)['estimated_duration']

# 使用示例
if __name__ == "__main__":
    estimator = init_estimator()
    print(estimate_duration('你好', estimator))

    # 测试用例
    test_cases = [
        # "Hello world this is a test",  # 纯英文
        # "你好世界 这是一个测试",      # 中文带空格
        # "Hello 你好 world 世界",      # 中英混合
        # "The weather is nice 所以我们去公园",  # 中英混合带空格
        # "我们需要在输出中体现空格的停顿时间",
        # "I couldn't help but notice the vibrant colors of the autumn leaves cascading gently from the trees"
        "가을 나뭇잎이 부드럽게 떨어지는 생생한 색깔을 주목하지 않을 수 없었다"
    ]
    
    for text in test_cases:
        result = estimator.process_mixed_text(text)
        print(f"\nText: {text}")
        print(f"Total syllables: {result['total_syllables']}")
        print(f"Estimated duration: {result['estimated_duration']:.2f}s")
        print("Language breakdown:")
        for lang, info in result['language_breakdown'].items():
            print(f"- {lang}: {info['syllables']} syllables ({info['text']})")
        print(f"Punctuation: {result['punctuation']}")
        print(f"Spaces: {result['spaces']}")
