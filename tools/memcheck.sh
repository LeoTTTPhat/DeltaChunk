#!/bin/bash
# Print current system memory use in GB (active+wired+compressed) and top 10 processes by RSS.
vm_stat | awk '/page size of/{gsub("[^0-9]","",$8); ps=$8} /Pages active/{a=$3} /Pages wired down/{w=$4} /Pages occupied by compressor/{c=$5} END{gsub("\\.","",a);gsub("\\.","",w);gsub("\\.","",c); printf "system used: %.1f GB (cap 200 GB)\n", (a+w+c)*ps/1073741824}'
ps -axo rss=,pid=,comm= | sort -rn | head -10 | awk '{printf "%8.1f GB  pid %-7s %s\n", $1/1048576, $2, $3}'
