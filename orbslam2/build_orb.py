
import os, sys, subprocess, glob, concurrent.futures as cf, time
WS = os.path.dirname(os.path.abspath(__file__))
MSVC=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
SDK=r"C:\Program Files (x86)\Windows Kits\10"; V="10.0.26100.0"
ENVROOT=r"C:\Users\ThinkPad\.claude-science\conda\envs\slam"
env=dict(os.environ)
env["INCLUDE"]=";".join([MSVC+r"\include",SDK+rf"\Include\{V}\ucrt",SDK+rf"\Include\{V}\um",SDK+rf"\Include\{V}\shared"])
env["LIB"]=";".join([MSVC+r"\lib\x64",SDK+rf"\Lib\{V}\ucrt\x64",SDK+rf"\Lib\{V}\um\x64"])
env["PATH"]=MSVC+r"\bin\Hostx64\x64;"+SDK+rf"\bin\{V}\x64;"+env["PATH"]
CL=MSVC+r"\bin\Hostx64\x64\cl.exe"; LIBEXE=MSVC+r"\bin\Hostx64\x64\lib.exe"; LINK=MSVC+r"\bin\Hostx64\x64\link.exe"
O=os.path.join(WS,"orb"); B=os.path.join(WS,"orb_obj"); os.makedirs(B,exist_ok=True)
OCV=os.path.join(WS,"ext","opencv","build")
INC=[OCV+r"\include", ENVROOT+r"\Library\include\eigen3", O, O+r"\include", O+r"\Thirdparty\g2o", O+r"\Thirdparty\DBoW2", O+r"\compat"]
FLAGS=["/nologo","/c","/O2","/EHsc","/MD","/bigobj","/W0","/utf-8","/std:c++14","/D_USE_MATH_DEFINES","/DNOMINMAX","/DCOMPILEDWITHC11","/D_CRT_SECURE_NO_WARNINGS","/DWINDOWS","/D_WINDOWS"]+["/I"+i for i in INC]
g2o=[l.strip() for l in open(os.path.join(O,"g2o_sources.txt"))]
groups={
 "DBoW2":[O+r"\Thirdparty\DBoW2\DBoW2\\"+f for f in ["BowVector.cpp","FORB.cpp","FeatureVector.cpp","ScoringObject.cpp"]]+[O+r"\Thirdparty\DBoW2\DUtils\Random.cpp",O+r"\Thirdparty\DBoW2\DUtils\Timestamp.cpp"],
 "g2o":[os.path.join(O,"Thirdparty","g2o",s.replace("/","\\")) for s in g2o],
 "ORB_SLAM2":sorted(glob.glob(O+r"\src\*.cc")),
}
hdr_t=max(os.path.getmtime(f) for f in glob.glob(O+r"\include\*.h")+glob.glob(O+r"\Thirdparty\**\*.h*",recursive=True))
def obj_of(src,g): return os.path.join(B,g+"__"+os.path.basename(src)+".obj")
def compile_one(args):
    src,g=args; obj=obj_of(src,g)
    if os.path.exists(obj) and os.path.getmtime(obj)>max(os.path.getmtime(src),hdr_t): return (src,0,"")
    lang=["/Tc"+src] if src.endswith(".c") else ["/Tp"+src]
    r=subprocess.run([CL]+FLAGS+["/Fo"+obj]+lang,env=env,capture_output=True,text=True,errors="replace")
    return (src,r.returncode,r.stdout+r.stderr)
def build_libs():
    jobs=[(s,g) for g,ss in groups.items() for s in ss]
    fails=[]
    with cf.ThreadPoolExecutor(8) as ex:
        for src,rc,out in ex.map(compile_one,jobs):
            if rc: fails.append((src,out))
    for src,out in fails:
        print("FAIL",os.path.basename(src)); print("\n".join([l.split(": ",1)[-1][:220] for l in out.splitlines() if "error" in l][:4]))
    if fails: return False
    for g,ss in groups.items():
        r=subprocess.run([LIBEXE,"/nologo","/OUT:"+os.path.join(B,g+".lib")]+[obj_of(s,g) for s in ss],env=env,capture_output=True,text=True,errors="replace")
        if r.returncode: print(r.stdout); return False
    return True
def build_exe(name, srcs, libs):
    objs=[]
    for s in srcs:
        src,rc,out=compile_one((s,"tool"))
        if rc: print("FAIL",s); print("\n".join([l for l in out.splitlines() if "error" in l][:10])); return False
        objs.append(obj_of(s,"tool"))
    exe=os.path.join(WS,"bin",name+".exe"); os.makedirs(os.path.dirname(exe),exist_ok=True)
    r=subprocess.run([LINK,"/nologo","/OUT:"+exe]+objs+[os.path.join(B,l+".lib") for l in libs]+[OCV+r"\x64\vc16\lib\opencv_world4100.lib"],env=env,capture_output=True,text=True,errors="replace")
    if r.returncode: print(r.stdout[-3000:]); return False
    return True
if __name__=="__main__":
    t=time.time(); ok=build_libs(); print("libs",ok, round(time.time()-t,1),"s")
    if ok:
        print("voc_convert", build_exe("voc_convert",[O+r"\tools\voc_convert.cc"],["DBoW2"]))
        print("run_seq", build_exe("run_seq",[O+r"\tools\run_seq.cc"],["ORB_SLAM2","g2o","DBoW2"]))
