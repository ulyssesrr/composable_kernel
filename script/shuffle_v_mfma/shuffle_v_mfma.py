import re

def read_example_asm_file(asm_file_path):
    with open(asm_file_path) as f:
        lines = f.readlines()
    return lines

def write_example_asm_file_back(asm_file_path, new_asm_txt):
    with open(asm_file_path, "w") as f:
        f.writelines(new_asm_txt)

class asm_file_analyser:
    def __init__(self, asm_txt):
        self.asm_txt = asm_txt
        self.core_loop_txt_bb0 = self.gen_core_loop_txt(".LBB0_1")
        self.core_loop_txt_bb1 = self.gen_core_loop_txt(".LBB1_1")

        self.next_free_vgpr = self.find_next_free_vgpr(asm_txt)

        self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb0 = self.enlarge_ds_read(self.core_loop_txt_bb0)
        self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb1 = self.enlarge_ds_read(self.core_loop_txt_bb1)

        self.inst_weight_dict_bb0 = self.gen_inst_weight(self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb0)
        self.inst_weight_dict_bb1 = self.gen_inst_weight(self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb1)

        self.interleave_vmfma_bb0, self.interleave_other_bb0 = self.get_interleave_start_line(self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb0)
        self.interleave_vmfma_bb1, self.interleave_other_bb1 = self.get_interleave_start_line(self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb1)

        self.reshuffle_inst_slot_bb0 = self.mfma_shuffle(self.interleave_vmfma_bb0, self.interleave_other_bb0, self.inst_weight_dict_bb0)
        self.reshuffle_inst_slot_bb1 = self.mfma_shuffle(self.interleave_vmfma_bb1, self.interleave_other_bb1, self.inst_weight_dict_bb1)

        self.new_asm_txt_bb0 = self.gen_new_asm_txt(self.interleave_vmfma_bb0, self.interleave_other_bb0, self.reshuffle_inst_slot_bb0, self.asm_txt, self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb0)
        
        for line in self.new_asm_txt_bb0:
            print(line)
        
        self.new_asm_txt_bb1 = self.gen_new_asm_txt(self.interleave_vmfma_bb1, self.interleave_other_bb1, self.reshuffle_inst_slot_bb1, self.new_asm_txt_bb0, self.core_loop_txt_bb0_enlarge_ds_read_vgpr_bb1)

    def gen_core_loop_txt(self, branch_name):
        core_loop = []
        loop_flag = 0
        for line in self.asm_txt:
            if line.find(f"{branch_name}:") != -1:
                loop_flag = 1
            if line.find(f"s_cbranch_scc1 {branch_name}") != -1:
                loop_flag = 0
            if loop_flag == 1:
                core_loop.append(line)

        return core_loop

    def find_next_free_vgpr(self, asm_txt):
        for line in asm_txt:
            numvpgr_str = re.findall(r'(?<=; NumVgprs: )\d*', line)
            if len(numvpgr_str) != 0:
                next_free_vgpr = int(numvpgr_str[0])
                print(next_free_vgpr)
                return next_free_vgpr

    def enlarge_ds_read(self, core_loop_txt):
        new_core_loop = []
        ds_read_list = []
        next_free_vgpr = self.next_free_vgpr
        
        vgpr_replacement_list = []
        for i in range(len(core_loop_txt)):
            line = core_loop_txt[i]
            find_ds_read = re.search(r'ds_read2_b64 v\[\d*:\d*\]', line)
            if find_ds_read:
                ds_read_line = find_ds_read.group()
                if ds_read_line in ds_read_list:
                    new_core_loop.append(re.sub(r'v\[\d*:\d*\]', f"v[{next_free_vgpr}:{next_free_vgpr+3}]", line))
                    old_vgpr_str = re.findall(r'(?<=v\[)\d+', line)
                    old_vgpr = int(old_vgpr_str[0])
                    vgpr_replacement_mfma = {}
                    vgpr_replacement_mfma[f"v[{old_vgpr}:{old_vgpr+1}]"] = f"v[{next_free_vgpr}:{next_free_vgpr+1}]"
                    vgpr_replacement_mfma[f"v[{old_vgpr+2}:{old_vgpr+3}]"] = f"v[{next_free_vgpr+2}:{next_free_vgpr+3}]"
                    vgpr_replacement_list.append([i, vgpr_replacement_mfma])
                    next_free_vgpr += 4
                else:
                    new_core_loop.append(line)
                    ds_read_list.append(find_ds_read.group())

            else:
                new_core_loop.append(line)

        #print(vgpr_replacement_list)

        core_loop_suf_vgpr = []
        for i in range(len(new_core_loop)):
            line = new_core_loop[i]
            if line.find("v_mfma") != -1:
                v_pair = re.findall(r'v\[\d*:\d*]', line)
                #print(i, v_pair)
                new_line = line
                for i_rep in vgpr_replacement_list:
                    if i > i_rep[0]:
                        if v_pair[0] in i_rep[1].keys():
                            new_line = new_line.replace(v_pair[0], i_rep[1][v_pair[0]])
                        if v_pair[1] in i_rep[1].keys():
                            new_line = new_line.replace(v_pair[1], i_rep[1][v_pair[1]])

                        #print(new_line)
                
                core_loop_suf_vgpr.append(new_line)
            else:
                core_loop_suf_vgpr.append(line)

        #print(vgpr_replacement_list)
        #for i in core_loop_suf_vgpr:
        #    print(i)
        return core_loop_suf_vgpr

    def gen_inst_weight(self, core_loop_txt):
        #core_loop_txt = self.core_loop_txt_bb0
        inst_weight_dict = {}
        for line in core_loop_txt:
            if line.find("ds_write2_b64") != -1:
                inst_weight_dict[line] = 30
            elif line.find("v_mul_lo_u32") != -1:
                inst_weight_dict[line] = 8
            elif line.find("v_mul_hi_u32") != -1:
                inst_weight_dict[line] = 8
            elif line.find("v_mfma") != -1:
                inst_weight_dict[line] = 56
            elif line.find("buffer_load_dword") != -1:
                inst_weight_dict[line] = 30
            elif line.find("s_barrier") != -1:
                inst_weight_dict[line] = 52
            elif line.find(";") != -1:
                inst_weight_dict[line] = 0
            elif len(line.strip()) == 0:
                inst_weight_dict[line] = 0
            else:
                inst_weight_dict[line] = 4

        return inst_weight_dict

    def get_interleave_start_line(self, core_loop_txt):
        #core_loop_txt = self.core_loop_txt_bb0
        vgpr_for_vmfma = []
        vgpr_for_others = []
        interleave_vmfma = []
        interleave_other = []

        i_last_ds_read = 0
        for i_inst in range(len(core_loop_txt)):
            line = core_loop_txt[i_inst]
            if line.find("ds_read") != -1:
                i_last_ds_read = i_inst

        i_last_ds_read += 1

        for i_inst in range(i_last_ds_read, len(core_loop_txt)):
            line = core_loop_txt[i_inst]
            if line.find("v_mfma") != -1:
                prev_line = core_loop_txt[i_inst - 1]
                if prev_line.find("s_waitcnt lgkmcnt(") != -1:
                    interleave_other.pop()
                    interleave_vmfma.append([prev_line, i_inst - 1])

                v_list_str = re.findall(r"(?<=v\[)\d*:\d*(?=\])", line)
                interleave_vmfma.append([line, i_inst])
                for v_str in v_list_str:
                    a = v_str.split(":")
                    vgpr_for_vmfma.append(int(a[0]))
                    vgpr_for_vmfma.append(int(a[1]))

            else:
                v_list_str = re.findall(r"(?<=v)\d+", line)
                
                if len(v_list_str) > 0:
                    for x in v_list_str:
                        if int(x) in vgpr_for_vmfma:
                            vgpr_for_vmfma=[]
                            interleave_vmfma = []
                            interleave_other = []

                interleave_other.append([line, i_inst])

        #self.interleave_vmfma = interleave_vmfma
        #self.interleave_other = interleave_other

        return interleave_vmfma, interleave_other

    def mfma_shuffle(self, interleave_vmfma, interleave_other, inst_weight_dict):
        reshuffle_inst_slot = []
        #interleave_vmfma = self.interleave_vmfma
        #interleave_other = self.interleave_other
        #inst_weight_dict = self.inst_weight_dict

        #print(inst_weight_dict)
        #return
        gap = 56
        i_inst_others = 0
        for vmfma in interleave_vmfma:
            tmp_gap = 0
            reshuffle_inst_slot.append(vmfma[0])
            if vmfma[0].find("s_waitcnt lgkmcnt(") != -1:
                pass
            else:
                while i_inst_others < len(interleave_other):
                    tmp_gap += inst_weight_dict[interleave_other[i_inst_others][0]]
                    #print(f"{interleave_other[i_inst_others][0][:-1]}: {inst_weight_dict[interleave_other[i_inst_others][0]]}")
                    #print(tmp_gap)
                    if tmp_gap > gap:
                        break
                    else:
                        reshuffle_inst_slot.append(interleave_other[i_inst_others][0])
                        i_inst_others += 1

        if i_inst_others < len(interleave_other):
            for i in range(i_inst_others, len(interleave_other)):
                reshuffle_inst_slot.append(interleave_other[i][0])

        #self.reshuffle_inst_slot = reshuffle_inst_slot
        return reshuffle_inst_slot

    def gen_new_asm_txt(self, interleave_vmfma, interleave_other, reshuffle_inst_slot, asm_txt, core_loop_txt):
        new_asm_txt = []
        new_core_loop = []
        update_for_loop = 0

        reshuffle_start_point = min(interleave_vmfma[0][1], interleave_other[0][1])

        for i in range(len(core_loop_txt)):
            if i >= reshuffle_start_point and (i - reshuffle_start_point) < len(reshuffle_inst_slot):
                new_core_loop.append(reshuffle_inst_slot[i - reshuffle_start_point])
            else:
                new_core_loop.append(core_loop_txt[i])

        for i in range(len(asm_txt)):
            if update_for_loop == 0:
                if asm_txt[i] == new_core_loop[0]:
                    update_for_loop = 1
            if len(new_core_loop) != 0 and update_for_loop:
                new_asm_txt.append(new_core_loop.pop(0))
            else:
                new_asm_txt.append(asm_txt[i])

        #self.new_asm_txt = new_asm_txt

        return new_asm_txt


if __name__ == "__main__":
    asm_path = "example/11_conv2d_bwd_weight/conv2d_bwd_weight_xdl-hip-amdgcn-amd-amdhsa-gfx908.s"
    asm_txt = read_example_asm_file(asm_path)
    asm_analyser = asm_file_analyser(asm_txt)
    #asm_analyser.get_interleave_start_line()
    #asm_analyser.mfma_shuffle()
    #asm_analyser.gen_new_asm_txt()
    #print(asm_analyser.inst_weight_dict)
    write_example_asm_file_back(asm_path, asm_analyser.new_asm_txt_bb1)