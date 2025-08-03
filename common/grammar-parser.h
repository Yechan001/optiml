// Implements a parser for an extended Backus-Naur form (BNF), producing the
// binary context-free grammar format specified by llama.h. Supports character
// ranges, grouping, and repetition operators. As an example, a grammar for
// arithmetic might look like:
//
// root  ::= expr
// expr  ::= term ([-+*/] term)*
// term  ::= num | "(" space expr ")" space
// num   ::= [0-9]+ space
// space ::= [ \t\n]*
/**
 * @file grammar-parser.h
 * @brief BNF-like grammar parser for constrained text generation
 * 
 * Implements a parser for grammar rules that can constrain
 * the model's output to follow specific patterns.
 */

#pragma once

#ifndef OPTIML_GRAMMAR_PARSER_H
#define OPTIML_GRAMMAR_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <string>
namespace grammar_parser {
    /**
     * @brief Grammar parse state
     * 
     * Contains the parsed representation of grammar rules
     * used for constrained generation.
     */
    struct parse_state {
        std::map<std::string, uint32_t>                 symbols; ///< Symbol table
        std::vector<std::vector<std::vector<uint32_t>>> rules;   ///< Grammar rules
        std::string                                     start;   ///< Starting rule
    };

    /**
     * @brief Parse grammar rules from source text
     * 
     * @param src Grammar rules in text format
     * @return parse_state Parsed grammar representation
     */
    parse_state parse(const char * src);
    void print_grammar(FILE * file, const parse_state & state);
}

#endif // OPTIML_GRAMMAR_PARSER_H
