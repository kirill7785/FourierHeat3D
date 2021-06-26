unit UnitLauncher;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.StdCtrls, Vcl.ExtCtrls;

type
  TFormLauncher = class(TForm)
    RadioGroup1: TRadioGroup;
    ButtonApply: TButton;
    ComboBoxGPUid: TComboBox;
    Label1: TLabel;
    procedure ButtonApplyClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  FormLauncher: TFormLauncher;

implementation

{$R *.dfm}

procedure TFormLauncher.ButtonApplyClick(Sender: TObject);
begin
   Close;
end;

end.
