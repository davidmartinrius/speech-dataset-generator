import pyphen

from pypinyin import pinyin, Style
from konlpy.tag import Okt

class SpeechRate:
            
    def check_language_availability(self, language):
        language_codes = list(set(code.split('_')[0] for code in pyphen.LANGUAGES.keys()))

        language_codes.extend(['zh','ko'])

        if language not in language_codes:
            raise Exception("Available language codes:", language_codes)
    
    def count_syllables_in_pinyin(self, pinyin_text):
        # Convert Pinyin to numbered Pinyin (with tone numbers)
        pinyin_with_tone_numbers = pinyin(pinyin_text, style=Style.TONE3)

        # Count the number of syllables
        syllable_count = sum([1 for s in pinyin_with_tone_numbers if s[0][-1].isdigit()])
        
        return syllable_count

    def get_total_syllables_per_word(self, word, language):
        
        self.check_language_availability(language)
        
        if 'zh' == language:
            
            pinyin_with_tone_numbers = pinyin(word, style=Style.TONE3)

            # Count the number of syllables
            total_syllables = sum([1 for s in pinyin_with_tone_numbers if s[0][-1].isdigit()])
        
        elif 'ko' == language:
            
            okt = Okt()
            morphemes = okt.morphs(word)
            total_syllables = len(morphemes)
        else:
            
            dic = pyphen.Pyphen(lang=language)
            total_syllables = len(dic.inserted(word).split('-'))
            
        return total_syllables

    def get_syllables_per_minute(self, words, language, duration_in_seconds):

        total_syllables = sum(self.get_total_syllables_per_word(word, language) for word in words)

        spm = (total_syllables / duration_in_seconds) * 60

        return round(spm, 3)
    
    def get_words_per_minute(self, words, duration_in_seconds):
        
        wpm = (len(words) / duration_in_seconds) * 60

        return round(wpm, 3)