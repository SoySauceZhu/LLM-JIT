for n in 0100 0500 9000; do
	  uv run --locked --extra lora python code/student_baseline.py \
		      --device cuda --split validation --prompt-version v3 \
		          --adapter "output/lora/gemma3-270m-it_v3_n${n}" \
			      --teacher-run output/baseline_v3/teacher_deepseek-v4.1-flash-1017_all \
			          --output "output/lora_eval/gemma3-270m-it_v3_n${n}_validation" \
				      --run || break
			      done
