cat /proc/sys/vm/overcommit_memory
echo "1" > /proc/sys/vm/overcommit_memory
cat /proc/sys/vm/overcommit_memory