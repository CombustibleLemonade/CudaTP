#include "parser.h"
#include <fstream>
#include <cassert>

#define ARITY 2

char capitalize(char s){
    return toupper((unsigned char)s);
}

int char_to_int(char x){
    switch (capitalize(x)) {
    case '0':
        return 0;
    case '1':
        return 1;
    case '2':
        return 2;
    case '3':
        return 3;
    case '4':
        return 4;
    case '5':
        return 5;
    case '6':
        return 6;
    case '7':
        return 7;
    case '8':
        return 8;
    case '9':
        return 9;
    case 'A':
        return 10;
    case 'B':
        return 11;
    case 'C':
        return 12;
    case 'D':
        return 13;
    case 'E':
        return 14;
    case 'F':
        return 15;
    }
    return -324098576;
}

int char_arr_to_int(int* text_idx, const char* text){
    int result = 0;

    for(int i = 2; i < 10; i++){
        result <<= 4;
        result += char_to_int(text[*text_idx + i]);
    }

    return result;
}

int parse_atom(const char* line, CudaTheorem* state, int* tei);

int parse_pair(const char* line, CudaTheorem* theorem, int* tei){
    int result_idx = theorem->pair_list.length++;
    if (theorem->pair_list.length >= CUDA_PAIR_LIST_LENGTH) {
        throw "Theorem too large!";
    }

    CudaPair* p = &theorem->pair_list.pair_list[result_idx];
    for(int i = 0; i < 2; i++){
        int atom = parse_atom(line, theorem, tei);
        if (atom == 0) throw;

        if (i == 0){
            p->a = atom;
            *tei += 2;
        } else {
            p->b = atom;
        }
    }
    
    return result_idx | PAIR_MASK;
}

int parse_atom(const char* line, CudaTheorem* state, int* tei){
    if(line[*tei] == '0'){
        int atom = char_arr_to_int(tei, line);
        *tei += 10;
        return atom;
    } else if (line[*tei] == '('){
        *tei += 1;
        int result = parse_pair(line, state, tei);
        assert(state->clause_list.length >= 0);
        *tei += 1;
        return result;
    } else if (line[*tei] == ','){
        return 0;
    } else if (line[*tei] == '\n'){
        return 0;
    } else if (line[*tei] == '\000'){
        return 0;
    }
    throw;
}

#include <iostream>

void parse_clause(const char* line, CudaTheorem* theorem){
    assert(theorem->clause_list.length >= 0);
    CudaClause* c = &theorem->clause_list.clauses[theorem->clause_list.length++];
    assert(theorem->clause_list.length < CLAUSE_COUNT);

    int* tei = (int*) malloc(sizeof(int));
    *tei = 0;

    for(int i = 0; i < CLAUSE_WIDTH; i++){

        
        CudaLiteral* l = &c->literals[i];
        c->length++;
        while (line[*tei] == ' '){
            *tei += 1;
        }

        if (line[*tei] == '-'){
            l->negative = true;
            *tei += 1;
        } else {
            l->negative = false;
        }

        l->node = parse_atom(line, theorem, tei);
        l->clause_idx = theorem->clause_list.length - 1;
        l->literal_idx = i;

        if (l->node == 0) {
            c->length -= 1;
        }

        *tei += 1;
    }
    if (c->length == 0) throw;

    free(tei);
}

void parse_file(const char* file_name, CudaTheorem* state){
    FILE *fp = fopen(file_name, "r");

    if (fp == NULL) {
        perror("Unable to open file");
        throw;
    }

    char* line = NULL;
    size_t len = 0;

    int l = 0;
    while(getline(&line, &len, fp) != -1) {
        l++;
        parse_clause(line, state);
        free(line);
        line = NULL;
        len = 0;
    }
    free(line);

    fclose(fp);
}

CudaTheorem* parse_file(const char* file_name){
    CudaTheorem* th = new CudaTheorem;
    parse_file(file_name, th);
    return th;
}