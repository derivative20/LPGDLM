#!/usr/bin/env python3
"""
Leixcal Extraction Script

This script processes JSON input files containing sign language video metadata
and extracts relevant data for training, development, or testing phases.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import thulac


class JSONValidator:
    """Validator for checking JSON file structure and content."""
    
    REQUIRED_FIELDS = {"name", "pose_path", "text"}
    
    @staticmethod
    def validate_entry(entry: Dict[str, Any], index: int) -> List[str]:
        """
        Validate a single JSON entry.
        
        Args:
            entry: The dictionary entry to validate
            index: The index of the entry for error reporting
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check if entry is a dictionary
        if not isinstance(entry, dict):
            errors.append(f"Entry {index}: Expected dictionary, got {type(entry).__name__}")
            return errors
        
        # Check for required fields
        missing_fields = JSONValidator.REQUIRED_FIELDS - set(entry.keys())
        if missing_fields:
            errors.append(f"Entry {index}: Missing required field(s): {', '.join(missing_fields)}")
        
        # Validate field types and content
        for field in JSONValidator.REQUIRED_FIELDS:
            if field in entry:
                if not isinstance(entry[field], str):
                    errors.append(f"Entry {index}: Field '{field}' must be a string, got {type(entry[field]).__name__}")
                elif not entry[field].strip():
                    errors.append(f"Entry {index}: Field '{field}' cannot be empty")
        
        return errors
    
    @staticmethod
    def validate_json(data: List[Dict[str, Any]]) -> bool:
        """
        Validate the entire JSON data structure.
        
        Args:
            data: The parsed JSON data
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, list):
            print(f"Error: JSON data must be a list, got {type(data).__name__}")
            return False
        
        if not data:
            print("Warning: JSON file is empty")
            return True
        
        has_errors = False
        for i, entry in enumerate(data):
            errors = JSONValidator.validate_entry(entry, i)
            if errors:
                has_errors = True
                for error in errors:
                    print(f"  {error}")
        
        return not has_errors


def load_json_file(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load and validate JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data if valid, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Successfully loaded JSON file: {file_path}")
        print(f"Total entries: {len(data)}")
        print("\nValidating JSON structure...")
        
        if JSONValidator.validate_json(data):
            print("✅ JSON validation passed!")
            return data
        else:
            print("❌ JSON validation failed!")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return None
    except FileNotFoundError:
        print(f"❌ Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error: Unexpected error loading file - {e}")
        return None


def process_data(data, args) -> None:
    """
    Process and extract data from the JSON entries.
    """

    os.makedirs(args.output_path, exist_ok=True)
    print(f"\nProcessing {args.phase} data...")
    print(f"Output directory: {args.output_path}")

    if args.phase != "train":
        ret_data = [{"name": sample['name'], "text": sample['text'], "pose": sample['pose_path']} for sample in data]
        return ret_data

    # load POS and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llada_tokenizer, trust_remote_code=True)
    lac = thulac.thulac()
    
    # Process each entry
    ret_data = []
    for sample in tqdm(data):
        text = sample['text']
        token_ids = tokenizer.encode(text)

        prev_text = ""
        tid_buff = []
        has_mask = False
        sample_ret = {"name": sample['name'], "text": sample['text'], "pose": sample['pose_path'], "mask_id": []}
        for i, tid in enumerate(token_ids):
            current_text = tokenizer.decode(token_ids[:i+1])
            new_part = current_text[len(prev_text):]
            
            tid_buff.append(token_ids[i])
            if new_part:
                # POS Tagger extraction
                anno_list = lac.cut(new_part)
                for anno in anno_list:
                    if anno[1] in ['r', 'n', 'v']:
                        has_mask = True
                        sample_ret['mask_id'].extend(tid_buff)
                        break
                tid_buff = []
            else:
                continue

            prev_text = current_text

            if not has_mask:
                # fallback to all masking
                sample_ret['mask_id'] = token_ids
            sample_ret['mask_id'] = list(set(sample_ret['mask_id']))
        ret_data.append(sample_ret)
    
    # Save extracted data
    output_file = os.path.join(args.output_path, f'{args.phase}_ready_{args.dataset}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ret_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed {len(ret_data)} entries")
    print(f"✅ Result saved to: {output_file}")


def main():
    """Main function to run the extraction script."""
    parser = argparse.ArgumentParser(
        description="Leixcal Extraction Script - Process sign language video metadata"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to the input JSON file containing the data entries'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to the output folder where results will be saved'
    )
    parser.add_argument(
        '--phase',
        type=str,
        required=True,
        choices=['train', 'dev', 'test'],
        help='The phase of the data in the dataset (train/dev/test)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        default="csl_daily",
        help='The name of the dataset'
    )
    parser.add_argument(
        '--llada_tokenizer',
        type=str,
        required=True,
        help='Path to the LLaDA weights folder (for tokenization)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Leixcal Extraction Script")
    print("=" * 60)
    print(f"Input JSON: {args.input_json}")
    print(f"Output Path: {args.output_path}")
    print(f"Phase: {args.phase}")
    print("-" * 60)
    
    # Load and validate JSON
    data = load_json_file(args.input_json)
    if data is None:
        sys.exit(1)
    
    # Process the data
    process_data(data, args)
    
    print("\n" + "=" * 60)
    print("✅ Extraction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()