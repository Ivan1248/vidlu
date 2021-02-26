declare -a servers=("strider.zemris.fer.hr" "celeborn" "treebeard" "magellan" "shelob" "nazgul")

for server in "${servers[@]}"; do
  echo $server $HOSTNAME
  ssh "$USER"@$server "nvidia-smi | sed '/^ *$/q'"
done