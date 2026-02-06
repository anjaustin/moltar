mkdir -p ~/.termux/boot
cat > ~/.termux/boot/start_trix.sh << 'INNER'
#!/data/data/com.termux/files/usr/bin/sh
HOME=/data/data/com.termux/files/home
export LD_LIBRARY_PATH=$HOME/noprofile
pkill -9 llama-server 2>/dev/null
sleep 1
nohup taskset 0x80 $HOME/noprofile/llama-server -m $HOME/LFM2-1.2B-Q4_0.gguf -t 1 --host 0.0.0.0 --port 8080 -c 3840 -np 1 --no-mmap --path $HOME/www > $HOME/server.log 2>&1 &
disown
INNER
chmod 755 ~/.termux/boot/start_trix.sh
~/.termux/boot/start_trix.sh
sleep 3
pgrep llama-server && echo "TriX running!"
