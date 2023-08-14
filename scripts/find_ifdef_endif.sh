#!/bin/bash

# 각 파일에 대해 조건을 확인
find . -name '*.cpp' -or -name '*.h' | while read file; do
    # 각 전처리문의 개수를 얻음

    if_count=$(grep -o '^[^/]*#if' "$file" | wc -l)
    endif_count=$(grep -o '^[^/]*#endif' "$file" | wc -l)

    # 전처리문의 총 합산
    total=$((if_count - endif_count))

    # 총 합산이 0이 아닌 경우 출력
    if [ $total -ne 0 ]; then
        echo "$file - #ifdef: $ifdef_count, #if: $if_count, #ifndef: $ifndef_count, #endif: $endif_count"
    fi
done
