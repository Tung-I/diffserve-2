# 2) OR kill anything listening on worker ports (50051–50066) and sink (50100)
for p in {50051..50066} 50100; do
  lsof -ti tcp:$p | xargs -r kill
done
sleep 2
for p in {50051..50066} 50100; do
  lsof -ti tcp:$p | xargs -r kill -9
done