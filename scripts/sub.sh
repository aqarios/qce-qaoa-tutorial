#!/usr/bin/env sh

jq '(.cells[] | select(has("source")) | .source) |= 
  (
    def process_tasks:
      . as $lines |
      (map(. | test("^[ \\t]*##TASK>")) | index(true)) as $start |
      if $start == null then
        .
      else
        (map(. | test("^[ \\t]*##<TASKEND")) | index(true)) as $end |
        if ($end != null and $end > $start) then
          ($lines[$start] | match("^([ \\t]*)") | .captures[0].string) as $indent |
          ($lines[:$start] + 
           [
             $indent + ("#") * (88 - ($indent|length)) + "\n",
             $indent + "### YOUR CODE BELOW" + "\n", 
             $indent + ("#") * (88 - ($indent|length)) + "\n",
             "\n",
             "\n",
             "\n",
             $indent + ("#") * (88 - ($indent|length)) + "\n",
             $indent + "### UNTIL HERE" + "\n",
             $indent + ("#") * (88 - ($indent|length)) + "\n"
           ] + 
           $lines[$end+1:]) | process_tasks
        else
          .
        end
      end;
    process_tasks
  )' $1
