#!/usr/bin/env bash

for i in {1..15}; do
  echo "Running instance $(printf %02d $i)..."
  time ./count "$(printf %02d $i)"
  echo ""
done
