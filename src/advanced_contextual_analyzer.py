#!/usr/bin/env python3
"""
Advanced Contextual Analyzer for PDF Structure Detection
Provides sophisticated document understanding and context analysis
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentContext:
    """Document context information"""
    total_pages: int
    total_blocks: int
    font_distribution: Dict[str, int]
    size_distribution: Dict[float, int]
    dominant_font: str
    dominant_size: float
    document_type: str
    language: str
    structure_complexity: float

class AdvancedContextualAnalyzer:
    """Advanced contextual analysis for better document understanding"""
    
    def __init__(self):
        # Document type patterns
        self.document_patterns = {
            'academic': [
                r'abstract\b', r'introduction\b', r'methodology\b', r'results\b',
                r'discussion\b', r'conclusion\b', r'references\b', r'bibliography\b',
                r'literature\s+review\b', r'related\s+work\b', r'experimental\b'
            ],
            'technical': [
                r'specification\b', r'requirements\b', r'implementation\b',
                r'architecture\b', r'design\b', r'system\b', r'manual\b',
                r'documentation\b', r'api\b', r'configuration\b', r'deployment\b'
            ],
            'business': [
                r'executive\s+summary\b', r'market\s+analysis\b', r'financial\b',
                r'revenue\b', r'strategy\b', r'objectives\b', r'stakeholders\b',
                r'timeline\b', r'budget\b', r'roi\b', r'kpi\b'
            ],
            'legal': [
                r'whereas\b', r'therefore\b', r'shall\b', r'pursuant\b',
                r'heretofore\b', r'hereinafter\b', r'notwithstanding\b',
                r'contract\b', r'agreement\b', r'terms\b', r'conditions\b'
            ],
            'manual': [
                r'installation\b', r'setup\b', r'configuration\b', r'troubleshooting\b',
                r'maintenance\b', r'operation\b', r'safety\b', r'warning\b',
                r'caution\b', r'step\s+\d+\b', r'procedure\b'
            ]
        }
        
        # Language detection patterns
        self.language_patterns = {
            'english': [
                r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
                r'\b(this|that|these|those|what|which|who|when|where|why|how)\b'
            ],
            'spanish': [
                r'\b(el|la|los|las|un|una|de|del|en|con|por|para|que|se|es)\b',
                r'\b(esto|eso|estos|esas|qué|cuál|quién|cuándo|dónde|por qué|cómo)\b'
            ],
            'french': [
                r'\b(le|la|les|un|une|de|du|des|en|dans|avec|pour|que|se|est)\b',
                r'\b(ce|cette|ces|ceux|quel|quelle|qui|quand|où|pourquoi|comment)\b'
            ],
            'german': [
                r'\b(der|die|das|den|dem|des|ein|eine|einen|einem|einer|und|oder)\b',
                r'\b(dies|das|diese|dieser|was|welche|wer|wann|wo|warum|wie)\b'
            ]
        }
        
        # Section ordering patterns
        self.section_orders = {
            'academic': [
                'abstract', 'introduction', 'background', 'related work',
                'methodology', 'method', 'approach', 'implementation',
                'results', 'findings', 'analysis', 'discussion',
                'conclusion', 'future work', 'acknowledgments', 'references'
            ],
            'technical': [
                'overview', 'introduction', 'requirements', 'architecture',
                'design', 'implementation', 'configuration', 'deployment',
                'testing', 'troubleshooting', 'maintenance', 'appendix'
            ],
            'business': [
                'executive summary', 'introduction', 'market analysis',
                'objectives', 'strategy', 'implementation', 'timeline',
                'budget', 'risks', 'recommendations', 'conclusion'
            ]
        }
        
        # Heading level indicators
        self.level_indicators = {
            'H1': [
                r'^\d+\.\s',  # 1. 2. 3.
                r'^[IVXLCDM]+\.\s',  # I. II. III.
                r'^chapter\s+\d+\b',  # Chapter 1
                r'^part\s+\d+\b',  # Part 1
                r'^section\s+\d+\b'  # Section 1
            ],
            'H2': [
                r'^\d+\.\d+\s',  # 1.1 1.2
                r'^\d+\.\d+\.\s',  # 1.1. 1.2.
                r'^[a-z]\)\s',  # a) b) c)
                r'^[A-Z]\)\s'  # A) B) C)
            ],
            'H3': [
                r'^\d+\.\d+\.\d+\s',  # 1.1.1 1.1.2
                r'^\([a-z]\)\s',  # (a) (b) (c)
                r'^\([A-Z]\)\s',  # (A) (B) (C)
                r'^\([0-9]+\)\s'  # (1) (2) (3)
            ]
        }
    
    def analyze_document_context(self, blocks: List[Dict[str, Any]]) -> DocumentContext:
        """Analyze overall document context"""
        
        if not blocks:
            return DocumentContext(
                total_pages=0, total_blocks=0, font_distribution={},
                size_distribution={}, dominant_font="", dominant_size=0,
                document_type="unknown", language="unknown", structure_complexity=0
            )
        
        # Basic statistics
        total_pages = max(block['page'] for block in blocks)
        total_blocks = len(blocks)
        
        # Font analysis
        font_counts = Counter(block['font_name'] for block in blocks)
        size_counts = Counter(block['font_size'] for block in blocks)
        
        dominant_font = font_counts.most_common(1)[0][0] if font_counts else ""
        dominant_size = size_counts.most_common(1)[0][0] if size_counts else 0
        
        # Document type detection
        document_type = self._detect_document_type(blocks)
        
        # Language detection
        language = self._detect_language(blocks)
        
        # Structure complexity
        structure_complexity = self._calculate_structure_complexity(blocks)
        
        return DocumentContext(
            total_pages=total_pages,
            total_blocks=total_blocks,
            font_distribution=dict(font_counts),
            size_distribution=dict(size_counts),
            dominant_font=dominant_font,
            dominant_size=dominant_size,
            document_type=document_type,
            language=language,
            structure_complexity=structure_complexity
        )
    
    def _detect_document_type(self, blocks: List[Dict[str, Any]]) -> str:
        """Detect document type based on content patterns"""
        
        # Combine all text
        all_text = ' '.join(block['text'].lower() for block in blocks)
        
        type_scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, all_text, re.IGNORECASE))
                score += matches
            
            # Normalize by document length
            type_scores[doc_type] = score / max(len(all_text.split()), 1)
        
        if not type_scores:
            return "unknown"
        
        return max(type_scores, key=type_scores.get)
    
    def _detect_language(self, blocks: List[Dict[str, Any]]) -> str:
        """Detect document language"""
        
        # Sample text from first few blocks
        sample_text = ' '.join(
            block['text'].lower() 
            for block in blocks[:min(10, len(blocks))]
        )
        
        if not sample_text:
            return "unknown"
        
        language_scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, sample_text, re.IGNORECASE))
                score += matches
            
            language_scores[lang] = score
        
        if not language_scores:
            return "unknown"
        
        return max(language_scores, key=language_scores.get)
    
    def _calculate_structure_complexity(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate document structure complexity"""
        
        if not blocks:
            return 0
        
        # Factors that contribute to complexity
        factors = []
        
        # 1. Font size diversity
        font_sizes = [block['font_size'] for block in blocks]
        size_diversity = len(set(font_sizes)) / len(font_sizes)
        factors.append(size_diversity)
        
        # 2. Number of pages
        pages = max(block['page'] for block in blocks)
        page_complexity = min(pages / 50, 1.0)  # Normalize to 50 pages
        factors.append(page_complexity)
        
        # 3. Number of potential headings
        potential_headings = sum(
            1 for block in blocks 
            if block.get('heading_likelihood', 0) > 0
        )
        heading_ratio = potential_headings / len(blocks)
        factors.append(heading_ratio)
        
        # 4. Text length variation
        text_lengths = [len(block['text']) for block in blocks]
        length_std = np.std(text_lengths) / max(np.mean(text_lengths), 1)
        length_complexity = min(length_std / 100, 1.0)
        factors.append(length_complexity)
        
        return sum(factors) / len(factors)
    
    def enhance_heading_detection(self, blocks: List[Dict[str, Any]], 
                                 context: DocumentContext) -> List[Dict[str, Any]]:
        """Enhance heading detection using contextual analysis"""
        
        # Sort blocks by position
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        # Detect section patterns
        section_patterns = self._detect_section_patterns(sorted_blocks, context)
        
        # Apply contextual rules
        enhanced_blocks = []
        
        for i, block in enumerate(sorted_blocks):
            enhanced_block = block.copy()
            
            # Apply document type specific rules
            enhanced_block = self._apply_document_type_rules(
                enhanced_block, context, i, sorted_blocks
            )
            
            # Apply section pattern rules
            enhanced_block = self._apply_section_pattern_rules(
                enhanced_block, section_patterns, i
            )
            
            # Apply sequential analysis
            enhanced_block = self._apply_sequential_analysis(
                enhanced_block, i, sorted_blocks
            )
            
            enhanced_blocks.append(enhanced_block)
        
        return enhanced_blocks
    
    def _detect_section_patterns(self, blocks: List[Dict[str, Any]], 
                                context: DocumentContext) -> Dict[str, Any]:
        """Detect common section patterns in the document"""
        
        patterns = {
            'numbered_sections': [],
            'lettered_sections': [],
            'keyword_sections': [],
            'indent_levels': defaultdict(list)
        }
        
        for i, block in enumerate(blocks):
            text = block['text'].strip()
            
            # Check for numbered sections
            if re.match(r'^\d+(\.\d+)*\.?\s', text):
                patterns['numbered_sections'].append(i)
            
            # Check for lettered sections
            if re.match(r'^[A-Za-z]\)\s', text):
                patterns['lettered_sections'].append(i)
            
            # Check for keyword sections
            if context.document_type in self.section_orders:
                expected_sections = self.section_orders[context.document_type]
                text_lower = text.lower()
                
                for section in expected_sections:
                    if section in text_lower:
                        patterns['keyword_sections'].append((i, section))
                        break
            
            # Check for indent levels
            indent_level = block['bbox'][0]  # X position as proxy for indent
            patterns['indent_levels'][indent_level].append(i)
        
        return patterns
    
    def _apply_document_type_rules(self, block: Dict[str, Any], 
                                  context: DocumentContext, 
                                  index: int, 
                                  all_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply document type specific rules"""
        
        text = block['text'].strip().lower()
        
        # Academic paper rules
        if context.document_type == 'academic':
            academic_sections = [
                'abstract', 'introduction', 'methodology', 'results',
                'discussion', 'conclusion', 'references'
            ]
            
            for section in academic_sections:
                if section in text:
                    block['contextual_h1_bonus'] = 3
                    break
        
        # Technical document rules
        elif context.document_type == 'technical':
            if any(word in text for word in ['api', 'configuration', 'installation']):
                block['contextual_h1_bonus'] = 2
            
            if re.match(r'^\d+\.\s', text):
                block['contextual_h1_bonus'] = 2
        
        # Business document rules
        elif context.document_type == 'business':
            if 'executive summary' in text:
                block['contextual_h1_bonus'] = 4
            elif any(word in text for word in ['objectives', 'strategy', 'analysis']):
                block['contextual_h1_bonus'] = 2
        
        # Legal document rules
        elif context.document_type == 'legal':
            if re.match(r'^\d+\.\s', text) or text.startswith('article'):
                block['contextual_h1_bonus'] = 2
        
        return block
    
    def _apply_section_pattern_rules(self, block: Dict[str, Any], 
                                    patterns: Dict[str, Any], 
                                    index: int) -> Dict[str, Any]:
        """Apply section pattern rules"""
        
        text = block['text'].strip()
        
        # Numbered section rules
        if index in patterns['numbered_sections']:
            depth = len(re.findall(r'\.', text.split()[0]))
            if depth == 0:  # 1. 2. 3.
                block['pattern_h1_bonus'] = 3
            elif depth == 1:  # 1.1 1.2
                block['pattern_h2_bonus'] = 3
            elif depth == 2:  # 1.1.1 1.1.2
                block['pattern_h3_bonus'] = 3
        
        # Lettered section rules
        if index in patterns['lettered_sections']:
            block['pattern_h2_bonus'] = 2
        
        # Keyword section rules
        for pattern_index, section in patterns['keyword_sections']:
            if index == pattern_index:
                block['pattern_h1_bonus'] = 2
                break
        
        return block
    
    def _apply_sequential_analysis(self, block: Dict[str, Any], 
                                  index: int, 
                                  all_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply sequential analysis rules"""
        
        # Look at surrounding blocks
        prev_block = all_blocks[index - 1] if index > 0 else None
        next_block = all_blocks[index + 1] if index < len(all_blocks) - 1 else None
        
        # If surrounded by long paragraphs, likely a heading
        if prev_block and next_block:
            prev_long = len(prev_block['text']) > 200
            next_long = len(next_block['text']) > 200
            current_short = len(block['text']) < 100
            
            if prev_long and next_long and current_short:
                block['sequential_heading_bonus'] = 2
        
        # If followed by several short blocks, might be a section header
        if index < len(all_blocks) - 3:
            following_short = all(
                len(all_blocks[i]['text']) < 50 
                for i in range(index + 1, min(index + 4, len(all_blocks)))
            )
            
            if following_short:
                block['sequential_list_bonus'] = 1
        
        return block
    
    def calculate_contextual_scores(self, blocks: List[Dict[str, Any]], 
                                   context: DocumentContext) -> List[Dict[str, Any]]:
        """Calculate enhanced contextual scores"""
        
        for block in blocks:
            # Start with existing scores
            title_score = block.get('title_score', 0)
            h1_score = block.get('h1_score', 0)
            h2_score = block.get('h2_score', 0)
            h3_score = block.get('h3_score', 0)
            
            # Add contextual bonuses
            h1_score += block.get('contextual_h1_bonus', 0)
            h1_score += block.get('pattern_h1_bonus', 0)
            h2_score += block.get('contextual_h2_bonus', 0)
            h2_score += block.get('pattern_h2_bonus', 0)
            h3_score += block.get('contextual_h3_bonus', 0)
            h3_score += block.get('pattern_h3_bonus', 0)
            
            # Add sequential bonuses
            heading_bonus = block.get('sequential_heading_bonus', 0)
            h1_score += heading_bonus
            h2_score += heading_bonus * 0.8
            h3_score += heading_bonus * 0.6
            
            # Update scores
            block['contextual_title_score'] = title_score
            block['contextual_h1_score'] = h1_score
            block['contextual_h2_score'] = h2_score
            block['contextual_h3_score'] = h3_score
            
            # Update overall heading likelihood
            block['contextual_heading_likelihood'] = max(
                title_score, h1_score, h2_score, h3_score
            )
        
        return blocks
    
    def validate_document_structure(self, blocks: List[Dict[str, Any]], 
                                   context: DocumentContext) -> List[Dict[str, Any]]:
        """Validate and fix document structure issues"""
        
        # Sort by position
        sorted_blocks = sorted(blocks, key=lambda x: (x['page'], x['bbox'][1]))
        
        # Check for logical heading hierarchy
        validated_blocks = self._validate_heading_hierarchy(sorted_blocks)
        
        # Check for missing title
        validated_blocks = self._validate_title_presence(validated_blocks, context)
        
        # Check for reasonable heading distribution
        validated_blocks = self._validate_heading_distribution(validated_blocks)
        
        return validated_blocks
    
    def _validate_heading_hierarchy(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix heading hierarchy"""
        
        prev_level = 0
        
        for block in blocks:
            predicted_label = block.get('predicted_label', 'Paragraph')
            
            if predicted_label in ['H1', 'H2', 'H3']:
                current_level = int(predicted_label[1])
                
                # Don't allow jumping more than one level
                if current_level > prev_level + 1:
                    # Demote to appropriate level
                    new_level = prev_level + 1
                    block['predicted_label'] = f'H{new_level}'
                    block['hierarchy_corrected'] = True
                    current_level = new_level
                
                prev_level = current_level
        
        return blocks
    
    def _validate_title_presence(self, blocks: List[Dict[str, Any]], 
                                context: DocumentContext) -> List[Dict[str, Any]]:
        """Ensure document has a title"""
        
        has_title = any(
            block.get('predicted_label') == 'Title' 
            for block in blocks
        )
        
        if not has_title and blocks:
            # Find best title candidate
            first_page_blocks = [b for b in blocks if b['page'] == 1]
            
            if first_page_blocks:
                # Look for block with highest title score
                best_candidate = max(
                    first_page_blocks,
                    key=lambda x: x.get('contextual_title_score', 0)
                )
                
                if best_candidate.get('contextual_title_score', 0) > 1:
                    best_candidate['predicted_label'] = 'Title'
                    best_candidate['title_promoted'] = True
        
        return blocks
    
    def _validate_heading_distribution(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate heading distribution across document"""
        
        headings = [
            block for block in blocks 
            if block.get('predicted_label') in ['H1', 'H2', 'H3']
        ]
        
        total_blocks = len(blocks)
        heading_ratio = len(headings) / max(total_blocks, 1)
        
        # If too many headings (>30%), demote some low-confidence ones
        if heading_ratio > 0.3:
            # Sort headings by confidence
            headings.sort(key=lambda x: x.get('confidence', 0))
            
            # Demote lowest confidence headings
            num_to_demote = int(len(headings) * 0.2)
            for i in range(num_to_demote):
                headings[i]['predicted_label'] = 'Paragraph'
                headings[i]['demoted_low_confidence'] = True
        
        # If too few headings (<2%), promote some high-scoring blocks
        elif heading_ratio < 0.02 and total_blocks > 10:
            non_headings = [
                block for block in blocks 
                if block.get('predicted_label') not in ['Title', 'H1', 'H2', 'H3']
            ]
            
            # Sort by heading likelihood
            non_headings.sort(
                key=lambda x: x.get('contextual_heading_likelihood', 0), 
                reverse=True
            )
            
            # Promote top candidates
            num_to_promote = min(3, len(non_headings))
            for i in range(num_to_promote):
                if non_headings[i].get('contextual_heading_likelihood', 0) > 2:
                    non_headings[i]['predicted_label'] = 'H1'
                    non_headings[i]['promoted_high_likelihood'] = True
        
        return blocks