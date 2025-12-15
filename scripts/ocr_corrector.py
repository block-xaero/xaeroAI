"""
Domain-aware OCR correction for whiteboard text.
"""

import json
from pathlib import Path

# Default config path
CONFIG_PATH = Path(__file__).parent.parent / "config" / "ocr_dictionary.json"

class OCRCorrector:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.terms = []
        self.min_length = 3
        self.max_distance_short = 1
        self.max_distance_long = 2
        self._load_config()
    
    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
            
            self.min_length = config.get("min_length", 3)
            self.max_distance_short = config.get("max_distance_short", 1)
            self.max_distance_long = config.get("max_distance_long", 2)
            
            # Flatten all term categories
            for category, terms in config.get("terms", {}).items():
                self.terms.extend([t.upper() for t in terms])
        else:
            print(f"Warning: Config not found at {self.config_path}, using defaults")
            self._use_defaults()
    
    def _use_defaults(self):
        self.terms = [
            "START", "END", "STOP", "YES", "NO", "ERROR", "VALID",
            "PROCESS", "INPUT", "OUTPUT", "DATA", "CACHE", "DB"
        ]
    
    def add_term(self, term):
        """Add a custom term"""
        upper = term.upper()
        if upper not in self.terms:
            self.terms.append(upper)
    
    def add_terms(self, terms):
        """Add multiple terms"""
        for t in terms:
            self.add_term(t)
    
    def correct(self, text):
        """Correct OCR text using domain dictionary"""
        if not text:
            return text
        
        text_upper = text.upper().strip()
        
        if len(text_upper) < self.min_length:
            return text
        
        # Exact match
        if text_upper in self.terms:
            return text_upper
        
        # Adaptive threshold
        max_dist = self.max_distance_short if len(text_upper) <= 4 else self.max_distance_long
        
        best_match = None
        best_score = float('inf')
        
        for term in self.terms:
            len_diff = abs(len(term) - len(text_upper))
            if len_diff > max_dist:
                continue
            
            dist = self._levenshtein(text_upper, term)
            score = dist + (len_diff * 0.5)
            
            if dist <= max_dist and score < best_score:
                best_score = score
                best_match = term
        
        return best_match if best_match else text
    
    @staticmethod
    def _levenshtein(s1, s2):
        if len(s1) < len(s2):
            return OCRCorrector._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(min(
                    prev_row[j + 1] + 1,
                    curr_row[j] + 1,
                    prev_row[j] + (c1 != c2)
                ))
            prev_row = curr_row
        
        return prev_row[-1]


# Global instance for convenience
_corrector = None

def get_corrector():
    global _corrector
    if _corrector is None:
        _corrector = OCRCorrector()
    return _corrector

def correct_ocr(text):
    """Convenience function"""
    return get_corrector().correct(text)

def add_terms(terms):
    """Convenience function"""
    get_corrector().add_terms(terms)
