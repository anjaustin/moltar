#!/system/bin/sh
echo '🤖 LFN350 CHAT - WORKING!'
echo '========================'
echo 'Commands: r, w, s, q'
echo ''

while true; do
    echo -n 'You: '
    read input

    if [ "$input" = 'q' ]; then
        echo '👋 Thanks for chatting with LFN350!'
        break
    fi

    echo -n 'LFN350: '

    if [ "$input" = 'r' ]; then
        echo 'Reflective recursion enables awareness through self-reference.'
    elif [ "$input" = 'w' ]; then
        echo 'I am LFN350, 350M parameter AI for Motorola testing.'
    elif [ "$input" = 's' ]; then
        echo '~50-100ms responses on MediaTek MT6855V hardware.'
    else
        echo 'Try r, w, s, or q'
    fi
    echo ''
done