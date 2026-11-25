
for n in 0100 0500 9000; do
	  uv run --locked --extra lora python code/student_lora.py \
		      --train-csv "data/lora/teacher_deepseek-v4.1-flash-1017_v3/history_n${n}.csv" \
		          --device cuda \
			      --output "output/lora/gemma3-270m-it_v3_n${n}" \
			          --run || break
			  done
